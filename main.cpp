#include <filePLY.h>
#include <helper.h>
#include <icp.h>
#include <ppf.h>
#include <util.h>
#include <xsimd/xsimd.hpp>

int main(int argc, char *argv[]) {
    std::cout << "float batch size:" << xsimd::simd_type<float>::size << std::endl;

    ppf::PointCloud model;
    ppf::readPLY(argv[ 1 ], model);
    ppf::PointCloud scene;
    ppf::readPLY(argv[ 2 ], scene);
    scene.viewPoint = {620, 100, 500};
    auto tmp        = model;
    // model.normal.clear();
    scene.normal.clear();

    {
        ppf::Timer    t("train model");
        ppf::Detector detector;
        detector.trainModel(model, 0.04f);
        detector.save("1.model");
    }

    std::vector<Eigen::Matrix4f> pose;
    std::vector<float>           score;
    ppf::MatchResult             result;
    {
        ppf::Timer    t("match scene");
        ppf::Detector detector;
        detector.load("1.model");
        detector.matchScene(scene, pose, score, 0.04f, 0.1f, 0.2f,
                            ppf::MatchParam{0.2, 35, 10, true, 0.5, 0, true, true, 15, 0.3},
                            &result);
    }

    for (int i = 0; i < pose.size(); i++) {
        std::cout << pose[ i ] << std::endl;
        std::cout << score[ i ] << std::endl;
        auto pc = ppf::transformPointCloud(tmp, pose[ i ]);
        ppf::writePLY(std::string("out") + std::to_string(i) + ".ply", pc);
    }

    ppf::writePLY("sampledScene.ply", result.sampledScene);
    ppf::writePLY("sampledKeypoint.ply", result.keyPoint);

    return 0;
}

int main2(int argc, char *argv[]) {
    ppf::PointCloud model;
    ppf::readPLY(argv[ 1 ], model);
    ppf::PointCloud scene;
    ppf::readPLY(argv[ 2 ], scene);
    ppf::PointCloud model2;
    ppf::readPLY(argv[ 3 ], model2);

    std::cout << "model point size:" << model.point.size()
              << "\nscene point size:" << scene.point.size() << std::endl;

    ppf::ICP               icp(ppf::ConvergenceCriteria(10, 1.5f, 1.2f, 3.5f, 0.0001f));
    ppf::ConvergenceResult result;
    {
        ppf::Timer t("icp");
        result = icp.regist(model, scene);
    }

    std::cout << "converged: " << result.converged << "\n"
              << "type: " << static_cast<int>(result.type) << "\n"
              << "mse: " << result.mse << "\n"
              << "convergeRate: " << result.convergeRate << "\n"
              << "iterations: " << result.iterations << "\n"
              << "inliner: " << result.inliner << "\n"
              << "pose: \n"
              << result.pose;

    if (result.converged) {
        auto pct = ppf::transformPointCloud(model2, result.pose);
        ppf::writePLY("out2.ply", pct);
    }

    return 0;
}

int main3(int argc, char *argv[]) {
    ppf::PointCloud model;
    ppf::readPLY(argv[ 1 ], model);
    model.viewPoint = {620, 100, 500};
    std::cout << "point size:" << model.point.size() << std::endl;
    model.normal.clear();
    ppf::KDTree kdtree(model.point);
    {
        ppf::Timer               t("compute normal");
        std::vector<std::size_t> indices(model.point.size());
        for (int i = 0; i < indices.size(); i++)
            indices[ i ] = i;
        int size = indices.size();
        ppf::estimateNormal(model, indices, kdtree, 10, true);
    }

    ppf::writePLY("normal.ply", model);
    return 0;
}
