#!/bin/bash

# for the develop branch this script bumps the Tensile version and hash and the rocBLAS version

OLD_TENSILE_VERSION="TENSILE_VERSION 4.42.0"
NEW_TENSILE_VERSION="TENSILE_VERSION 4.43.0"

OLD_TENSILE_HASH="1e25e645f7f60c8edbe55ff1cf789e0daf2468e6"
NEW_TENSILE_HASH="f5dc5728610361dc8cc9aab9177a4462b122753b"

OLD_ROCBLAS_VERSION="4.4.0"
NEW_ROCBLAS_VERSION="4.5.0"

OLD_SO_VERSION="rocblas_SOVERSION 4.4"
NEW_SO_VERSION="rocblas_SOVERSION 4.5"

OLD_HIPBLASLT_VERSION="HIPBLASLT_VERSION 0.10"
NEW_HIPBLASLT_VERSION="HIPBLASLT_VERSION 0.12"

sed -i "s/${OLD_TENSILE_VERSION}/${NEW_TENSILE_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_TENSILE_HASH}/${NEW_TENSILE_HASH}/g" tensile_tag.txt

sed -i "s/${OLD_ROCBLAS_VERSION}/${NEW_ROCBLAS_VERSION}/g" CMakeLists.txt

sed -i "s/${OLD_SO_VERSION}/${NEW_SO_VERSION}/g" library/CMakeLists.txt

sed -i "s/${OLD_HIPBLASLT_VERSION}/${NEW_HIPBLASLT_VERSION}/g" CMakeLists.txt
