#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

const int WIDTH = 800;
const int HEIGHT = 600;

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    vk::UniqueInstance instance;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        // NOTE: instance destruction is handled by UniqueInstance

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance() {
        auto appInfo = vk::ApplicationInfo(
            "Hello Triangle",
            VK_MAKE_VERSION(1, 0, 0),
            "No Engine",
            VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_0
        );
        
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        auto createInfo = vk::InstanceCreateInfo(
            vk::InstanceCreateFlags(),
            &appInfo,
            0, nullptr, // enabled layers
            glfwExtensionCount, glfwExtensions // enabled extensions
        );

        try {
            instance = vk::createInstanceUnique(createInfo, nullptr);
        }
        catch (vk::SystemError err) {
            throw std::runtime_error("failed to create instance!");
        }

        std::cout << "available extensions:" << std::endl;

        for (const auto& extension : vk::enumerateInstanceExtensionProperties()) {
            std::cout << "\t" << extension.extensionName << std::endl;
        }
    }

};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
