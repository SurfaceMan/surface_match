#include "filePLY.h"
#include <rply.h>

#include <Eigen/Geometry>
#include <iostream>
#include <privateType.h>

namespace ppf {

namespace {

namespace ply_trianglemesh_reader {

struct PLYReaderState {
    PointCloud *mesh_ptr;
    long        vertex_index;
    long        vertex_num;
    long        normal_index;
    long        normal_num;
    long        color_index;
    long        color_num;
    long        face_index;
    long        face_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long            index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr), &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num)
        return 0;

    double value = ply_get_argument_value(argument);
    switch (index) {
        case 0: {
            state_ptr->mesh_ptr->point.x[ state_ptr->vertex_index ] = value;
            break;
        }
        case 1: {
            state_ptr->mesh_ptr->point.y[ state_ptr->vertex_index ] = value;
            break;
        }
        case 2: {
            state_ptr->mesh_ptr->point.z[ state_ptr->vertex_index ] = value;
            state_ptr->vertex_index++;
            break;
        }
    }

    return 1;
}

int ReadNormalCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long            index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr), &index);
    if (state_ptr->normal_index >= state_ptr->normal_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    switch (index) {
        case 0: {
            state_ptr->mesh_ptr->normal.x[ state_ptr->vertex_index ] = value;
            break;
        }
        case 1: {
            state_ptr->mesh_ptr->normal.y[ state_ptr->vertex_index ] = value;
            break;
        }
        case 2: {
            state_ptr->mesh_ptr->normal.z[ state_ptr->vertex_index ] = value;
            state_ptr->normal_index++;
            break;
        }
    }

    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long            index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr), &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    switch (index) {
        case 0: {
            state_ptr->mesh_ptr->color.x[ state_ptr->vertex_index ] = value;
            break;
        }
        case 1: {
            state_ptr->mesh_ptr->color.y[ state_ptr->vertex_index ] = value;
            break;
        }
        case 2: {
            state_ptr->mesh_ptr->color.z[ state_ptr->vertex_index ] = value;
            state_ptr->color_index++;
            break;
        }
    }
    return 1;
}

int ReadFaceCallBack(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long            length, index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr), &index);
    if (state_ptr->face_index >= state_ptr->face_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    ply_get_argument_property(argument, NULL, &length, &index);

    switch (index) {
        case 0: {
            state_ptr->mesh_ptr->face.x[ state_ptr->face_index ] = value;
            break;
        }
        case 1: {
            state_ptr->mesh_ptr->face.y[ state_ptr->face_index ] = value;
            break;
        }
        case 2: {
            state_ptr->mesh_ptr->face.z[ state_ptr->face_index ] = value;
            state_ptr->face_index++;
            break;
        }
    }

    return 1;
}

} // namespace ply_trianglemesh_reader
} // unnamed namespace
/// @endcond

bool PointCloud_ReadPLY(const char *filename, PointCloud_t *mesh) {
    using namespace ply_trianglemesh_reader;

    if (nullptr == mesh)
        return false;

    p_ply ply_file = ply_open(filename, NULL, 0, NULL);
    if (!ply_file) {
        std::cout << "Read PLY failed: unable to open file: " << filename << std::endl;
        return false;
    }
    if (!ply_read_header(ply_file)) {
        std::cout << "Read PLY failed: unable to parse header." << std::endl;
        ply_close(ply_file);
        return false;
    }

    if (*mesh == nullptr)
        *mesh = new PointCloud;

    PLYReaderState state;
    state.mesh_ptr   = *mesh;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x", ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx", ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red", ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        std::cout << "Read PLY failed: number of vertex <= 0." << std::endl;
        ply_close(ply_file);
        return false;
    }

    state.face_num =
        ply_set_read_cb(ply_file, "face", "vertex_indices", ReadFaceCallBack, &state, 0);
    if (state.face_num == 0) {
        state.face_num =
            ply_set_read_cb(ply_file, "face", "vertex_index", ReadFaceCallBack, &state, 0);
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index  = 0;
    state.face_index   = 0;

    (*mesh)->point.resize(state.vertex_num);
    (*mesh)->normal.resize(state.normal_num);
    (*mesh)->face.resize(state.face_num);
    (*mesh)->color.resize(state.color_num);

    if (!ply_read(ply_file)) {
        std::cout << "Read PLY failed: unable to read file: " << filename << std::endl;
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    return true;
}

bool PointCloud_WritePLY(const char *filename, const PointCloud_t mesh, bool write_ascii) {
    if (mesh == nullptr || mesh->empty()) {
        std::cout << "Write PLY failed: mesh has 0 vertices." << std::endl;
        return false;
    }

    p_ply ply_file =
        ply_create(filename, write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN, NULL, 0, NULL);
    if (!ply_file) {
        std::cout << "Write PLY failed: unable to open file: " << filename << std::endl;
        return false;
    }

    auto write_vertex_normals = mesh->hasNormal();
    auto write_vertex_colors  = !mesh->color.empty();

    ply_add_comment(ply_file, "Created by SurfaceMatch");
    ply_add_element(ply_file, "vertex", static_cast<long>(mesh->point.size()));
    ply_add_property(ply_file, "x", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    ply_add_property(ply_file, "y", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    ply_add_property(ply_file, "z", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    if (write_vertex_normals) {
        ply_add_property(ply_file, "nx", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
        ply_add_property(ply_file, "ny", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
        ply_add_property(ply_file, "nz", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    }
    if (write_vertex_colors) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    ply_add_element(ply_file, "face", static_cast<long>(mesh->face.size()));
    ply_add_property(ply_file, "vertex_indices", PLY_LIST, PLY_UCHAR, PLY_UINT);
    if (!ply_write_header(ply_file)) {
        std::cout << "Write PLY failed: unable to write header." << std::endl;
        ply_close(ply_file);
        return false;
    }

    bool printed_color_warning = false;
    for (size_t i = 0; i < mesh->point.size(); i++) {
        const auto &vertex = mesh->point[ i ];
        ply_write(ply_file, vertex.x());
        ply_write(ply_file, vertex.y());
        ply_write(ply_file, vertex.z());
        if (write_vertex_normals) {
            const auto &normal = mesh->normal[ i ];
            ply_write(ply_file, normal.x());
            ply_write(ply_file, normal.y());
            ply_write(ply_file, normal.z());
        }
        if (write_vertex_colors) {
            const auto &color = mesh->color[ i ];
            ply_write(ply_file, color.x());
            ply_write(ply_file, color.y());
            ply_write(ply_file, color.z());
        }
    }
    for (size_t i = 0; i < mesh->face.size(); i++) {
        const auto &triangle = mesh->face[ i ];
        ply_write(ply_file, 3);
        ply_write(ply_file, triangle.x());
        ply_write(ply_file, triangle.y());
        ply_write(ply_file, triangle.z());
    }

    ply_close(ply_file);
    return true;
}

} // namespace ppf
