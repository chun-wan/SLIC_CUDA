ROCM_PATH=/opt/rocm
SURFACE_FUNC=${ROCM_PATH}/hip/include/hip/amd_detail/amd_surface_functions.h
sed -i  "s/*data = 0;/*data = {0,0,0,0};/g" ${SURFACE_FUNC}
