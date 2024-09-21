#include "Data/TreeData.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "Driver/Device.h"
#include "nlohmann/json.hpp"

#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

using json = nlohmann::json;

namespace neutron {

class TreeData::Tree {
public:
  struct TreeNode : public Context {
    inline static const size_t RootId = 0;
    inline static const size_t DummyId = std::numeric_limits<size_t>::max();

    TreeNode() = default;
    explicit TreeNode(size_t id, const std::string &name)
        : id(id), Context(name) {}
    TreeNode(size_t id, size_t parentId, const std::string &name)
        : id(id), parentId(parentId), Context(name) {}
    virtual ~TreeNode() = default;

    void addChild(const Context &context, size_t id) { children[context] = id; }

    bool hasChild(const Context &context) const {
      return children.find(context) != children.end();
    }

    size_t getChild(const Context &context) const {
      return children.at(context);
    }

    size_t parentId = DummyId;
    size_t id = DummyId;
    std::map<Context, size_t> children = {};
    std::map<MetricKind, std::shared_ptr<Metric>> metrics = {};
    std::map<std::string, FlexibleMetric> flexibleMetrics = {};
    friend class Tree;
  };

  Tree() {
    treeNodeMap.try_emplace(TreeNode::RootId, TreeNode::RootId, "ROOT");
  }

  size_t addNode(const Context &context, size_t parentId) {
    if (treeNodeMap[parentId].hasChild(context)) {
      return treeNodeMap[parentId].getChild(context);
    }
    auto id = nextContextId++;
    treeNodeMap.try_emplace(id, id, parentId, context.name);
    treeNodeMap[parentId].addChild(context, id);
    return id;
  }

  size_t addNode(const std::vector<Context> &indices) {
    auto parentId = TreeNode::RootId;
    for (auto index : indices) {
      parentId = addNode(index, parentId);
    }
    return parentId;
  }

  TreeNode &getNode(size_t id) { return treeNodeMap.at(id); }

  enum class WalkPolicy { PreOrder, PostOrder };

  template <WalkPolicy walkPolicy, typename FnT> void walk(FnT &&fn) {
    if constexpr (walkPolicy == WalkPolicy::PreOrder) {
      walkPreOrder(TreeNode::RootId, fn);
    } else if constexpr (walkPolicy == WalkPolicy::PostOrder) {
      walkPostOrder(TreeNode::RootId, fn);
    }
  }

  template <typename FnT> void walkPreOrder(size_t contextId, FnT &&fn) {
    fn(getNode(contextId));
    for (auto &child : getNode(contextId).children) {
      walkPreOrder(child.second, fn);
    }
  }

  template <typename FnT> void walkPostOrder(size_t contextId, FnT &&fn) {
    for (auto &child : getNode(contextId).children) {
      walkPostOrder(child.second, fn);
    }
    fn(getNode(contextId));
  }

private:
  size_t nextContextId = TreeNode::RootId + 1;
  // tree node id->tree node
  std::map<size_t, TreeNode> treeNodeMap;
};

void TreeData::init() { tree = std::make_unique<Tree>(); }

void TreeData::startOp(const Scope &scope) {
  // enterOp and addMetric maybe called from different threads
  std::unique_lock<std::shared_mutex> lock(mutex);
  std::vector<Context> contexts;
  if (contextSource != nullptr)
    contexts = contextSource->getContexts();
  contexts.push_back(Context(scope.name));
  auto contextId = tree->addNode(contexts);
  scopeIdToContextId[scope.scopeId] = contextId;
}

void TreeData::stopOp(const Scope &scope) {}

size_t TreeData::addScope(size_t parentScopeId, const std::string &name) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(parentScopeId);
  auto scopeId = parentScopeId;
  if (scopeIdIt == scopeIdToContextId.end()) {
    std::vector<Context> contexts;
    if (contextSource != nullptr)
      contexts = contextSource->getContexts();
    // Record the parent context
    scopeIdToContextId[parentScopeId] = tree->addNode(contexts);
  } else {
    // Add a new context under it and update the context
    scopeId = Scope::getNewScopeId();
    scopeIdToContextId[scopeId] =
        tree->addNode(Context(name), scopeIdIt->second);
  }
  return scopeId;
}

void TreeData::addMetric(size_t scopeId, std::shared_ptr<Metric> metric) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  // The profile data is deactived, ignore the metric
  if (scopeIdIt == scopeIdToContextId.end())
    return;
  auto contextId = scopeIdIt->second;
  auto &node = tree->getNode(contextId);
  if (node.metrics.find(metric->getKind()) == node.metrics.end())
    node.metrics.emplace(metric->getKind(), metric);
  else
    node.metrics[metric->getKind()]->updateMetric(*metric);
}

void TreeData::addMetrics(size_t scopeId,
                          const std::map<std::string, MetricValueType> &metrics,
                          bool aggregable) {
  std::unique_lock<std::shared_mutex> lock(mutex);
  auto scopeIdIt = scopeIdToContextId.find(scopeId);
  auto contextId = Tree::TreeNode::DummyId;
  if (scopeIdIt == scopeIdToContextId.end()) {
    if (contextSource == nullptr)
      throw std::runtime_error("ContextSource is not set");
    // Attribute the metric to the last context
    std::vector<Context> contexts = contextSource->getContexts();
    contextId = tree->addNode(contexts);
  } else {
    contextId = scopeIdIt->second;
  }
  auto &node = tree->getNode(contextId);
  for (auto [metricName, metricValue] : metrics) {
    if (node.flexibleMetrics.find(metricName) == node.flexibleMetrics.end())
      node.flexibleMetrics.emplace(
          metricName, FlexibleMetric(metricName, metricValue, aggregable));
    else {
      node.flexibleMetrics.at(metricName).updateValue(metricValue);
    }
  }
}

void TreeData::dumpHatchet(std::ostream &os) const {
  std::map<size_t, json *> jsonNodes;
  json output = json::array();
  output.push_back(json::object());
  jsonNodes[Tree::TreeNode::RootId] = &(output.back());
  std::set<std::string> valueNames;
  std::map<uint64_t, std::set<uint64_t>> deviceIds;
  this->tree->template walk<Tree::WalkPolicy::PreOrder>(
      [&](Tree::TreeNode &treeNode) {
        const auto contextName = treeNode.name;
        auto contextId = treeNode.id;
        json *jsonNode = jsonNodes[contextId];
        (*jsonNode)["frame"] = {{"name", contextName}, {"type", "function"}};
        (*jsonNode)["metrics"] = json::object();
        for (auto [metricKind, metric] : treeNode.metrics) {
          if (metricKind == MetricKind::Kernel) {
            auto kernelMetric = std::dynamic_pointer_cast<KernelMetric>(metric);
            auto duration = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Duration));
            auto invocations = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::Invocations));
            auto deviceId = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceId));
            auto deviceType = std::get<uint64_t>(
                kernelMetric->getValue(KernelMetric::DeviceType));
            auto deviceTypeName =
                getDeviceTypeString(static_cast<DeviceType>(deviceType));
            (*jsonNode)["metrics"]
                       [kernelMetric->getValueName(KernelMetric::Duration)] =
                           duration;
            (*jsonNode)["metrics"]
                       [kernelMetric->getValueName(KernelMetric::Invocations)] =
                           invocations;
            (*jsonNode)["metrics"]
                       [kernelMetric->getValueName(KernelMetric::DeviceId)] =
                           std::to_string(deviceId);
            (*jsonNode)["metrics"]
                       [kernelMetric->getValueName(KernelMetric::DeviceType)] =
                           deviceTypeName;
            valueNames.insert(
                kernelMetric->getValueName(KernelMetric::Duration));
            valueNames.insert(
                kernelMetric->getValueName(KernelMetric::Invocations));
            deviceIds.insert({deviceType, {deviceId}});
          } else {
            throw std::runtime_error("MetricKind not supported");
          }
        }
        for (auto [_, flexibleMetric] : treeNode.flexibleMetrics) {
          auto valueName = flexibleMetric.getValueName(0);
          valueNames.insert(valueName);
          std::visit(
              [&](auto &&value) { (*jsonNode)["metrics"][valueName] = value; },
              flexibleMetric.getValues()[0]);
        }
        (*jsonNode)["children"] = json::array();
        auto children = treeNode.children;
        for (auto _ : children) {
          (*jsonNode)["children"].push_back(json::object());
        }
        auto idx = 0;
        for (auto child : children) {
          auto [index, childId] = child;
          jsonNodes[childId] = &(*jsonNode)["children"][idx];
          idx++;
        }
      });
  // Hints for all available metrics
  for (auto valueName : valueNames) {
    output[Tree::TreeNode::RootId]["metrics"][valueName] = 0;
  }
  // Prepare the device information
  // Note that this is done from the application thread,
  // query device information from the tool thread (e.g., CUPTI) will have
  // problems
  output.push_back(json::object());
  auto &deviceJson = output.back();
  for (auto [deviceType, deviceIds] : deviceIds) {
    auto deviceTypeName =
        getDeviceTypeString(static_cast<DeviceType>(deviceType));
    if (!deviceJson.contains(deviceTypeName))
      deviceJson[deviceTypeName] = json::object();
    for (auto deviceId : deviceIds) {
      Device device = getDevice(static_cast<DeviceType>(deviceType), deviceId);
      deviceJson[deviceTypeName][std::to_string(deviceId)] = {
          {"clock_rate", device.clockRate},
          {"memory_clock_rate", device.memoryClockRate},
          {"bus_width", device.busWidth},
          {"arch", device.arch},
          {"num_sms", device.numSms}};
    }
  }
  os << std::endl << output.dump(4) << std::endl;
}

void TreeData::doDump(std::ostream &os, OutputFormat outputFormat) const {
  std::shared_lock<std::shared_mutex> lock(mutex);
  if (outputFormat == OutputFormat::Hatchet) {
    dumpHatchet(os);
  } else {
    std::logic_error("OutputFormat not supported");
  }
}

TreeData::TreeData(const std::string &path, ContextSource *contextSource)
    : Data(path, contextSource) {
  init();
}

TreeData::~TreeData() {}

} // namespace neutron
