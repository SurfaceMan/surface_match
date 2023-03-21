/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011-16 Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#pragma once

#include <nanoflann.hpp>
#include <privateType.h>
#include <vector>

namespace ppf {

template <class Distance = nanoflann::metric_L2> struct KDTreeVector3FAdaptor {
    const static int dims = 3;
    using self_t          = KDTreeVector3FAdaptor<Distance>;
    using metric_t        = typename Distance::template traits<float, self_t>::distance_t;
    using index_t         = nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, dims>;

    /** The kd-tree index for the user to call its methods as usual with any
     * other FLANN index */
    index_t *index = nullptr;

    /// Constructor: takes a const ref to the vector of vectors object with the
    /// data points
    KDTreeVector3FAdaptor(const Vector3F &mat, const int leaf_max_size = 10,
                          const BoundingBox &box = {}, const VectorI &validIndices = {})
        : m_data(mat)
        , m_box(box)
        , m_validIndices(validIndices) {
        assert(mat.size() != 0);
        index = new index_t(static_cast<int>(dims), *this /* adaptor */,
                            nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
        // index->buildIndex();
    }

    ~KDTreeVector3FAdaptor() {
        delete index;
    }

    const Vector3F    &m_data;
    const BoundingBox &m_box;
    const VectorI     &m_validIndices;

    VectorI vAccTmp;

    /** Query for the \a num_closest closest points to a given point
     *  (entered as query_point[0:dim-1]).
     *  Note that this is a short-cut method for index->findNeighbors().
     *  The user can also call index->... methods as desired.
     *
     * \note nChecks_IGNORED is ignored but kept for compatibility with
     * the original FLANN interface.
     */
    inline void query(const float *query_point, const size_t num_closest, uint32_t *out_indices,
                      float *out_distances_sq, const int nChecks_IGNORED = 10) const {
        nanoflann::KNNResultSet<float, uint32_t> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point, nanoflann::SearchParams());
    }

    /** @name Interface expected by KDTreeSingleIndexAdaptor
     * @{ */

    const self_t &derived() const {
        return *this;
    }
    self_t &derived() {
        return *this;
    }

    const VectorI kdtree_init_indices() const {
        return m_validIndices;
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return m_validIndices.empty() ? m_data.size() : m_validIndices.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        switch (dim) {
            case 0:
                return m_data.x[ idx ];
            case 1:
                return m_data.y[ idx ];
            case 2:
                return m_data.z[ idx ];
        }
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    // Return true if the BBOX was already computed by the class and returned
    // in "bb" so it can be avoided to redo it again. Look at bb.size() to
    // find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX> bool kdtree_get_bbox(BBOX &bb) const {
        if (m_box._diameter == 0)
            return false;

        for (int i = 0; i < 3; i++) {
            bb[ i ].low  = m_box.min()[ i ];
            bb[ i ].high = m_box.max()[ i ];
        }
        return true;
    }

    void reduce(VectorI indices) {
        if (indices.size() != index->vAcc.size())
            throw std::runtime_error("Unmatched indices size");
        vAccTmp     = std::move(index->vAcc);
        index->vAcc = std::move(indices);
    }

    void restore() {
        if (vAccTmp.empty())
            return;
        index->vAcc = std::move(vAccTmp);
    }
};

using KDTree = KDTreeVector3FAdaptor<nanoflann::metric_L2>;
} // namespace ppf
