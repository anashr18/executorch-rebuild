rm -rf out/host
CC=$(command -v clang) CXX=$(command -v clang++) \
cmake -S . -B out/host -G Ninja \
  -DET_BUILD_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_AR=$(command -v llvm-ar || echo /usr/bin/ar) \
  -DCMAKE_RANLIB=$(command -v llvm-ranlib || echo /usr/bin/ranlib) \
  -DCMAKE_NM=$(command -v llvm-nm || echo /usr/bin/nm)
cmake --build out/host -j


grep -E 'CMAKE_(C|CXX)_COMPILER[:=]|CMAKE_(AR|RANLIB|NM)[:=]' out/host/CMakeCache.txt
# Expect clang/clang++ for COMPILER; llvm-ar/ranlib/nm (or GNU ar/ranlib/nm) for the rest



# Remove stale cache entries
sed -i '/CMAKE_\(C\|CXX\)_COMPILER_AR/d;/CMAKE_\(C\|CXX\)_COMPILER_RANLIB/d;/CMAKE_\(AR\|RANLIB\|NM\)/d' out/host/CMakeCache.txt
# Or simply rm -rf out/host and re-run the cmake command above.






