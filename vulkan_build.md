# submodule init
git submodule sync                                   
git submodule update --init --recursive       

# remove older/failed build files if any
rm -rf cmake-android-out  

# setup the vulkan SDK
download the linux tar file 
https://vulkan.lunarg.com/sdk/home
curl -o vulkansdk-linux-x86_64-1.4.328.1.tar.xz https://sdk.lunarg.com/sdk/download/1.4.328.1/linux/vulkansdk-linux-x86_64-1.4.328.1.tar.xz
cd ~ && mkdir -p vulkan_sdk  && cd vulkan_sdk  
tar -xvf vulkansdk-linux-x86_64-1.4.328.1.tar.xz
source ~/vulkan_sdk/1.4.328.1/setup-env.sh   

# cmake init 
cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DGLSLC_PATH=$(which glslc) \
    -Bcmake-android-out 
    
# cmake buid and install
cmake --build cmake-android-out -j64 --target install