# Getting Started with CUDA
In order to use the OpenCV CUDA modules to leverage an NVIDIA graphics card, you must compile opencv from its source files. Below are the instructions to follow to do this:

## Step-by-Step Installation Guide for Windows:
### 1. Install Required Software
- CMake: Download and install CMake from cmake.org.
- NVIDIA CUDA Toolkit: Download and install the CUDA Toolkit from NVIDIA's website. Make sure it matches your GPU and OS version.
- NVIDIA cuDNN: Download cuDNN from NVIDIA's website and install it by copying the files into the CUDA Toolkit installation directory (e.g., C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x).
- Visual Studio: Install Visual Studio (2019 or 2022) with the Desktop development with C++ workload. Visual Studio Community is free and works fine.

### 2. Download and Organize OpenCV Source Code
 - Download source code from OpenCV and OpenCV-Contrib repositories on GitHub.
   - From the branch dropdown in one of the repositories, click on the *Tags* tab and select the newest stable version. Download the zip file of this repository.
   - Repeat for the other repository, ensuring that the version number is the same.
   - Within you user on the local machine, create a new empty directory named something like "opencv_gpu".
   - Unzip the two repository files into this new directory.
   - within the opencv_gpu directory, create a new directory called "build".
  
### 3. Configure the Build with CMake
- Open CMake GUI (cmake-gui).
- Set the source code location: Point this to the directory where you extracted the OpenCV source code.
- Set where to build the binaries: Point this to a new directory where the build files will be generated (e.g., C:\opencv_build).- Click "Configure":
   - Choose "Visual Studio 16 2019" or "Visual Studio 17 2022" as the generator, depending on your Visual Studio version.
   - Choose the "x64" architecture.
 
- Once the inital configurations are done, we need to change a few of the values that show up highlighted in red.
   - In the CMake GUI, check the box for following options:
      - WITH_CUDA
      - OPENCV_DNN_CUDA
   - Set the OPENCV_EXTRA_MODULES_PATH to point to the *modules* folder inside the opencv_contrib directory.

- Click "Generate" to generate the Visual Studio solution files.

### 4. Build OpenCV with Visual Studio
- Open the generated solution file:
   - Navigate to the build directory (e.g., C:\opencv_build) and open the OpenCV.sln file in Visual Studio.
- Set build type:
   - In Visual Studio, set the build type to Release for better performance. To do this right-click on the solution explorer (where it says "Solution "OpenCV") and click on "Configuration Manager..."

- Build the project:
   - Right-click on the solution in the Solution Explorer and select "Build Solution". This step may take some time, depending on your system's specifications.
