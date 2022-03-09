#include <helper.h>
#include <icp.h>
#include <ppf.h>
#include <util.h>

int main1(int argc, char *argv[]) {
    auto model = ppf::loadText(argv[ 1 ]);
    auto scene = ppf::loadText(argv[ 2 ]);

    ppf::Detector detector;
    {
        ppf::Timer t("train model");
        detector.trainModel(model, 0.04f);
    }

    std::vector<Eigen::Matrix4f> pose;
    std::vector<float>           score;
    ppf::MatchResult             result;
    {
        ppf::Timer t("match scene");
        detector.matchScene(scene, pose, score, 0.04f, 0.1f, 0.6f, ppf::MatchParam{0.2, 5},
                            &result);
    }

    for (int i = 0; i < pose.size(); i++) {
        auto pc = ppf::transformPointCloud(model, pose[ i ]);
        ppf::saveText(std::string("out") + std::to_string(i) + ".txt", pc);

        std::cout << pose[ i ] << std::endl;
        std::cout << score[ i ] << std::endl;
    }

    ppf::saveText("sampledScene.txt", result.sampledScene);
    ppf::saveText("sampledKeypoint.txt", result.keyPoint);

    return 0;
}

int main2(int argc, char *argv[]) {
    auto model  = ppf::loadText(argv[ 1 ]);
    auto scene  = ppf::loadText(argv[ 2 ]);
    auto model2 = ppf::loadText(argv[ 3 ]);

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
        ppf::saveText("out2.txt", pct);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    auto model = ppf::loadText(argv[ 1 ]);

    ppf::KDTree kdtree(model.point);
    // auto        indices = ppf::findEdge(kdtree, model, 0.005f, 3.1415926f / 2.f);
    std::vector<std::size_t> indices;
    {
        ppf::Timer t("find edge");
        indices = ppf::findEdge(kdtree, model, 10);
    }

    ppf::PointCloud edges = ppf::extraIndices(model, indices);
    ppf::saveText("edges.txt", edges);
    return 0;
}