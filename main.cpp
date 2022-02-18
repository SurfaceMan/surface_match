#include <helper.h>
#include <ppf.h>
#include <util.h>

int main(int argc, char *argv[]) {
    auto model = ppf::loadText(argv[ 1 ]);
    auto scene = ppf::loadText(argv[ 2 ]);

    ppf::Detector detector;
    {
        ppf::Timer t("train model");
        detector.trainModel(model);
    }

    std::vector<Eigen::Matrix4f> pose;
    std::vector<float>           score;
    {
        ppf::Timer t("match scene");
        detector.matchScene(scene, pose, score);
    }

    auto pc = ppf::transformPointCloud(model, pose[ 0 ]);
    ppf::saveText("out.txt", pc);

    std::cout << pose[ 0 ] << std::endl;
    std::cout << score[ 0 ] << std::endl;
}