#include <helper.h>
#include <icp.h>
#include <ppf.h>
#include <util.h>

int main2(int argc, char *argv[]) {
    auto model = ppf::loadText(argv[ 1 ]);
    auto scene = ppf::loadText(argv[ 2 ]);

    ppf::Detector detector;
    {
        ppf::Timer t("train model");
        detector.trainModel(model, 0.04);
    }

    std::vector<Eigen::Matrix4f> pose;
    std::vector<float>           score;
    {
        ppf::Timer t("match scene");
        detector.matchScene(scene, pose, score, 0.04);
    }

    auto pc = ppf::transformPointCloud(model, pose[ 0 ]);
    ppf::saveText("out.txt", pc);

    std::cout << pose[ 0 ] << std::endl;
    std::cout << score[ 0 ] << std::endl;
}

int main(int argc, char *argv[]) {
    auto model  = ppf::loadText(argv[ 1 ]);
    auto scene  = ppf::loadText(argv[ 2 ]);
    auto model2 = ppf::loadText(argv[ 3 ]);

    std::cout << "model point size:" << model.point.size()
              << "scene point size:" << scene.point.size() << std::endl;

    ppf::ICP        icp(10);
    float           residual;
    Eigen::Matrix4f pose;
    {
        ppf::Timer t("icp");
        icp.registerModelToScene(model, scene, residual, pose);
    }
    auto pct = ppf::transformPointCloud(model2, pose);
    ppf::saveText("out2.txt", pct);

    std::cout << residual << std::endl << pose << std::endl;
    return 0;
}