project(Neutron CXX)

set(NEUTRON_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
file(GLOB_RECURSE NEUTRON_SRC ${NEUTRON_SRC_DIR}/lib/*.cpp)
add_library(neutron SHARED ${NEUTRON_SRC} ${NEUTRON_SRC_DIR}/${PROJECT_NAME}.cpp)

include_directories(${NEUTRON_SRC_DIR}/include)

if(PYTHON_INCLUDE_DIRS)
  # We have PYTHON_INCLUDE_DIRS set--this is what we expect when building
  # using pip install.
  include_directories(${PYTHON_INCLUDE_DIRS})
  include_directories(${PYBIND11_INCLUDE_DIR})
else()
  # Otherwise, we might be building from top CMakeLists.txt directly.
  # Try to find Python and pybind11 packages.
  find_package(Python3 REQUIRED Interpreter Development)
  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
  include_directories(${Python3_INCLUDE_DIRS})
  include_directories(${pybind11_INCLUDE_DIR})
endif()

# Check if the platform is MacOS
if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  # Other platforms build with -flto, but we found that this adds significant overhead to our macos CI without providing a major benefit.
  set(NEUTRON_PYTHON_LDFLAGS "-undefined dynamic_lookup")
endif()

target_compile_definitions(neutron PRIVATE __HIP_PLATFORM_AMD__)

target_link_libraries(neutron PRIVATE ${Python_LIBRARIES} ${NEUTRON_PYTHON_LDFLAGS})
