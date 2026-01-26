# 1. DEEP CLEAN: Remove conflicting local pip-installed build tools
# This is the most likely cause of the 'mesonbuild.options' error
pip uninstall -y meson ninja
rm -f /home/ubuntu/.local/bin/meson /home/ubuntu/.local/bin/ninja

# 2. Ensure system-wide build tools are present
sudo apt-get update
sudo apt-get install -y meson ninja-build libfftw3-dev libsndfile1-dev

# 3. Reset the build directory
# This is crucial because 'builddir' caches the path to the old meson
rm -rf builddir

# 4. Re-configure using the system-wide meson
# Verify we are using /usr/bin/meson by checking which
echo "Using meson at: $(which meson)"
meson setup builddir --prefix=/usr/local -Ddefault_library=both

# 5. Compile and Install
ninja -C builddir
sudo ninja -C builddir install

# 6. Refresh library cache
sudo ldconfig

# 7. Verify the binary is working
rubberband --version