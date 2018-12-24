# vulkan-tutorial-hpp
Following vulkan-tutorial.com using Vulkan-Hpp

## Raw notes:
- Vulkan SDK 1.1.92.1 from https://vulkan.lunarg.com/
- `.\vcpkg.exe install glm:x64-windows glfw3:x64-windows` (Hint: can be set permanently: `VCPKG_DEFAULT_TRIPLET=x64-windows`)
  - `.\vcpkg.exe integrate install` does not work for x64 apparently -> had to set additional include directories and linker settings manually
