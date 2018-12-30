# vulkan-tutorial-hpp
Following [vulkan-tutorial.com](https://vulkan-tutorial.com/) using [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp).

`main.cpp` contains the latest state and the [steps](/steps) folder contains the code for individual steps and diffs between them (`previous`) and between the Vulkan-Hpp version and the [original](https://github.com/Overv/VulkanTutorial/tree/master/code) (`original`).

## Notes:
- using Vulkan SDK 1.1.92.1
- dependency setup can be simplified with [vcpk](https://github.com/Microsoft/vcpkg)
  - `.\vcpkg.exe install glm:x64-windows glfw3:x64-windows` (Hint: can be set permanently: `VCPKG_DEFAULT_TRIPLET=x64-windows`)
  - `.\vcpkg.exe integrate install` does not work for x64 apparently -> had to set additional include directories and linker settings manually in Visual Studio
