#include <helper.h>
#include <icp.h>
#include <ppf.h>
#include <util.h>

int main(int argc, char *argv[]) {
    auto model = ppf::loadText(argv[ 1 ]);
    auto scene = ppf::loadText(argv[ 2 ]);

    ppf::Detector detector;
    {
        ppf::Timer t("train model");
        detector.trainModel(model, 0.04f);
    }

    std::vector<Eigen::Matrix4f> pose;
    std::vector<float>           score;
    {
        ppf::Timer t("match scene");
        detector.matchScene(scene, pose, score, 0.04f, 0.2f, 0.5f, 1);
    }

    for (int i = 0; i < pose.size(); i++) {
        auto pc = ppf::transformPointCloud(model, pose[ i ]);
        ppf::saveText(std::string("out") + std::to_string(i) + ".txt", pc);

        std::cout << pose[ i ] << std::endl;
        std::cout << score[ i ] << std::endl;
    }

    return 0;
}

int main2(int argc, char *argv[]) {
    auto model  = ppf::loadText(argv[ 1 ]);
    auto scene  = ppf::loadText(argv[ 2 ]);
    auto model2 = ppf::loadText(argv[ 3 ]);

    std::cout << "model point size:" << model.point.size()
              << "\nscene point size:" << scene.point.size() << std::endl;

    ppf::ICP               icp(ppf::ConvergenceCriteria(10, 1.5, 1.2, 3.5, 0.0001));
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