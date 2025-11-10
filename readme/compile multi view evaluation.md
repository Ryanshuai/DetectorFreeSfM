# Runbook: Build ETH3DMultiViewEvaluation for DetectorFreeSfM

## 0. Environment

* **OS**: Ubuntu 22.04
* **Project Root**: `~/code/DetectorFreeSfM`
* **Conda Env**: `detectorfreesfm`
* **Compiler**: Use **system gcc/g++**, not conda’s toolchain
* **Expected binary**: `DetectorFreeSfM/third_party/multi-view-evaluation/build/ETH3DMultiViewEvaluation`

---

## 1. Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    libeigen3-dev \
    libboost-all-dev \
    libpcl-dev
```

If you previously added an invalid PCL PPA (e.g. `v-launchpad-jochen-sprickerhof-de/pcl`), remove it:

```bash
sudo add-apt-repository --remove ppa:v-launchpad-jochen-sprickerhof-de/pcl || true
sudo rm /etc/apt/sources.list.d/*pcl*jamm*.list 2>/dev/null || true
sudo apt-get update
```

---

## 2. Clone Source

```bash
cd ~/code/DetectorFreeSfM
mkdir -p third_party && cd third_party

git clone https://github.com/ETH3D/multi-view-evaluation.git
cd multi-view-evaluation
```

---

## 3. Fix C++ Standard Conflict

Remove `-std=c++11` from `CMakeLists.txt`:

```bash
sed -i 's/ -std=c++11//g' CMakeLists.txt
```

Resulting section should look like:

```cmake
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  add_definitions("-O2 -msse2 -msse3")
endif()
```

---

## 4. Locate PCL CMake Config

Check where `PCLConfig.cmake` is:

```bash
dpkg -L libpcl-dev | grep -E "PCLConfig.cmake|pcl-config.cmake"
```

Usually found at:

```
/usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake
```

Use its directory as `PCL_DIR`.

---

## 5. Configure and Build

```bash
cd ~/code/DetectorFreeSfM/third_party/multi-view-evaluation
rm -rf build && mkdir build && cd build

cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DPCL_DIR=/usr/lib/x86_64-linux-gnu/cmake/pcl \
  -DBoost_INCLUDE_DIR=/usr/include \
  -DBoost_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu

make -j"$(nproc)"
```

Expect output:

```
[100%] Built target ETH3DMultiViewEvaluation
```

---

## 6. Verify

```bash
cd ~/code/DetectorFreeSfM/third_party/multi-view-evaluation/build
./ETH3DMultiViewEvaluation --help
```

If you see a warning like:

```
libtiff.so.5: no version information available ...
```

→ Ignore it. It’s just conda’s TIFF library mismatched with system VTK, does **not** affect results.

---

## 7. Integration with DetectorFreeSfM

Ensure your evaluation script points to the compiled binary:

```python
from pathlib import Path
EVAL_TOOL = Path("third_party/multi-view-evaluation/build/ETH3DMultiViewEvaluation")
```

Then run evaluation, e.g.:

```bash
cd ~/code/DetectorFreeSfM
python eval_dataset.py +eth3d_sfm=dfsfm.yaml neuralsfm.NEUSFM_coarse_matcher='loftr_official'
```

---

## 8. Minimal Steps for Future Rebuilds

If you rebuild on a new system:

1. `apt-get install -y cmake build-essential libeigen3-dev libboost-all-dev libpcl-dev`
2. Clone repo, delete `-std=c++11` from `CMakeLists.txt`
3. `cmake ..` with system gcc/g++ and correct `PCL_DIR`
4. `make -j$(nproc)`
5. Confirm `--help` runs successfully
