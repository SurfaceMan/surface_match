#include "filePLY.h"
#include <rply.h>

#include <Eigen/Geometry>
#include <iostream>

namespace ppf {

namespace {

bool IsPointInsidePolygon(const Eigen::MatrixX2f &polygon, double x, double y) {
    bool inside = false;
    for (int i = 0; i < polygon.rows(); ++i) {
        // i and j are the indices of the first and second vertices.
        int j = (i + 1) % polygon.rows();

        // The vertices of the edge that will be checked.
        double vx0 = polygon(i, 0);
        double vy0 = polygon(i, 1);
        double vx1 = polygon(j, 0);
        double vy1 = polygon(j, 1);

        // Check whether the edge intersects a line from (-inf,y) to (x,y).
        // First, check if the line crosses the horizontal line at y in either
        // direction.
        if (((vy0 <= y) && (vy1 > y)) || ((vy1 <= y) && (vy0 > y))) {
            // If so, get the point where it crosses that line.
            double cross = (vx1 - vx0) * (y - vy0) / (vy1 - vy0) + vx0;

            // Finally, check if it crosses to the left of the test point.
            if (cross < x)
                inside = !inside;
        }
    }
    return inside;
}

bool AddTrianglesByEarClipping(ppf::PointCloud &mesh, std::vector<unsigned int> &indices) {
    int             n           = int(indices.size());
    Eigen::Vector3f face_normal = Eigen::Vector3f::Zero();
    if (n > 3) {
        for (int i = 0; i < n; i++) {
            const Eigen::Vector3f &v1 =
                mesh.point[ indices[ (i + 1) % n ] ] - mesh.point[ indices[ i % n ] ];
            const Eigen::Vector3f &v2 =
                mesh.point[ indices[ (i + 2) % n ] ] - mesh.point[ indices[ (i + 1) % n ] ];
            face_normal += v1.cross(v2);
        }
        float l = std::sqrt(face_normal.dot(face_normal));
        face_normal *= (1.0 / l);
    }

    bool found_ear = true;
    while (n > 3) {
        if (!found_ear) {
            // If no ear is found after all indices are looped through, the
            // polygon is not triangulable.
            return false;
        }

        found_ear = false;
        for (int i = 1; i < n - 1; i++) {
            const Eigen::Vector3f &v1 = mesh.point[ indices[ i ] ] - mesh.point[ indices[ i - 1 ] ];
            const Eigen::Vector3f &v2 = mesh.point[ indices[ i + 1 ] ] - mesh.point[ indices[ i ] ];
            bool                   is_convex = (face_normal.dot(v1.cross(v2)) > 0.0);
            bool                   is_ear    = true;
            if (is_convex) {
                // If convex, check if vertex is an ear
                // (no vertices within triangle v[i-1], v[i], v[i+1])
                Eigen::MatrixX2f polygon(3, 2);
                for (int j = 0; j < 3; j++) {
                    polygon(j, 0) = mesh.point[ indices[ i + j - 1 ] ](0);
                    polygon(j, 1) = mesh.point[ indices[ i + j - 1 ] ](1);
                }

                for (int j = 0; j < n; j++) {
                    if (j == i - 1 || j == i || j == i + 1) {
                        continue;
                    }
                    const Eigen::Vector3f &v = mesh.point[ indices[ j ] ];
                    if (IsPointInsidePolygon(polygon, v(0), v(1))) {
                        is_ear = false;
                        break;
                    }
                }

                if (is_ear) {
                    found_ear = true;
                    mesh.face.push_back(
                        Eigen::Vector3i(indices[ i - 1 ], indices[ i ], indices[ i + 1 ]));
                    indices.erase(indices.begin() + i);
                    n = int(indices.size());
                    break;
                }
            }
        }
    }
    mesh.face.push_back(Eigen::Vector3i(indices[ 0 ], indices[ 1 ], indices[ 2 ]));

    return true;
}

namespace ply_trianglemesh_reader {

struct PLYReaderState {
    PointCloud               *mesh_ptr;
    long                      vertex_index;
    long                      vertex_num;
    long                      normal_index;
    long                      normal_num;
    long                      color_index;
    long                      color_num;
    std::vector<unsigned int> face;
    long                      face_index;
    long                      face_num;
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long            index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr), &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->point[ state_ptr->vertex_index ][ index ] = value;
    if (index == 2) { // reading 'z'
        state_ptr->vertex_index++;
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
    state_ptr->mesh_ptr->normal[ state_ptr->normal_index ][ index ] = value;
    if (index == 2) { // reading 'nz'
        state_ptr->normal_index++;
    }
    return 1;
}

/*
int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long            index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr), &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->mesh_ptr->vertex_colors_[ state_ptr->color_index ](index) = value / 255.0;
    if (index == 2) { // reading 'blue'
        state_ptr->color_index++;
    }
    return 1;
}
*/

int ReadFaceCallBack(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long            dummy, length, index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr), &dummy);
    double value = ply_get_argument_value(argument);
    if (state_ptr->face_index >= state_ptr->face_num) {
        return 0;
    }

    ply_get_argument_property(argument, NULL, &length, &index);
    if (index == -1) {
        state_ptr->face.clear();
    } else {
        state_ptr->face.push_back(int(value));
    }
    if (long(state_ptr->face.size()) == length) {
        if (!AddTrianglesByEarClipping(*state_ptr->mesh_ptr, state_ptr->face)) {
            std::cout << "Read PLY failed: A polygon in the mesh could not be "
                         "decomposed into triangles."
                      << std::endl;
            return 0;
        }
        state_ptr->face_index++;
    }
    return 1;
}

} // namespace ply_trianglemesh_reader
} // unnamed namespace
/// @endcond

bool readPLY(const std::string &filename, ppf::PointCloud &mesh) {
    using namespace ply_trianglemesh_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        std::cout << "Read PLY failed: unable to open file: " << filename << std::endl;
        return false;
    }
    if (!ply_read_header(ply_file)) {
        std::cout << "Read PLY failed: unable to parse header." << std::endl;
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.mesh_ptr   = &mesh;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x", ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx", ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    // state.color_num = ply_set_read_cb(ply_file, "vertex", "red", ReadColorCallback, &state, 0);
    // ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    // ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

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

    mesh.point.resize(state.vertex_num);
    mesh.normal.resize(state.normal_num);
    // mesh.vertex_colors_.resize(state.color_num);

    if (!ply_read(ply_file)) {
        std::cout << "Read PLY failed: unable to read file: " << filename << std::endl;
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    return true;
}

bool writePLY(const std::string &filename, const ppf::PointCloud &mesh, bool write_ascii) {
    if (mesh.empty()) {
        std::cout << "Write PLY failed: mesh has 0 vertices." << std::endl;
        return false;
    }

    p_ply ply_file =
        ply_create(filename.c_str(), write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN, NULL, 0, NULL);
    if (!ply_file) {
        std::cout << "Write PLY failed: unable to open file: " << filename << std::endl;
        return false;
    }

    auto write_vertex_normals = mesh.hasNormal();
    // write_vertex_colors  = write_vertex_colors && mesh.HasVertexColors();

    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "vertex", static_cast<long>(mesh.point.size()));
    ply_add_property(ply_file, "x", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    ply_add_property(ply_file, "y", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    ply_add_property(ply_file, "z", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    if (write_vertex_normals) {
        ply_add_property(ply_file, "nx", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
        ply_add_property(ply_file, "ny", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
        ply_add_property(ply_file, "nz", PLY_FLOAT, PLY_FLOAT, PLY_FLOAT);
    }
    // if (write_vertex_colors) {
    //     ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    //     ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    //     ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    // }
    ply_add_element(ply_file, "face", static_cast<long>(mesh.face.size()));
    ply_add_property(ply_file, "vertex_indices", PLY_LIST, PLY_UCHAR, PLY_UINT);
    if (!ply_write_header(ply_file)) {
        std::cout << "Write PLY failed: unable to write header." << std::endl;
        ply_close(ply_file);
        return false;
    }

    bool printed_color_warning = false;
    for (size_t i = 0; i < mesh.point.size(); i++) {
        const auto &vertex = mesh.point[ i ];
        ply_write(ply_file, vertex(0));
        ply_write(ply_file, vertex(1));
        ply_write(ply_file, vertex(2));
        if (write_vertex_normals) {
            const auto &normal = mesh.normal[ i ];
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        // if (write_vertex_colors) {
        //     const auto &color = mesh.vertex_colors_[ i ];
        //     if (!printed_color_warning && (color(0) < 0 || color(0) > 1 || color(1) < 0 ||
        //                                    color(1) > 1 || color(2) < 0 || color(2) > 1)) {
        //         utility::LogWarning("Write Ply clamped color value to valid range");
        //         printed_color_warning = true;
        //     }
        //     auto rgb = utility::ColorToUint8(color);
        //     ply_write(ply_file, rgb(0));
        //     ply_write(ply_file, rgb(1));
        //     ply_write(ply_file, rgb(2));
        // }
    }
    for (size_t i = 0; i < mesh.face.size(); i++) {
        const auto &triangle = mesh.face[ i ];
        ply_write(ply_file, 3);
        ply_write(ply_file, triangle(0));
        ply_write(ply_file, triangle(1));
        ply_write(ply_file, triangle(2));
    }

    ply_close(ply_file);
    return true;
}

} // namespace ppf
