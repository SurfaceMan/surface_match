#include <filePLY.h>
#include <helper.h>
#include <ppf.h>
#include <util.h>

int main(int argc, char *argv[]) {

    if (argc < 3)
        throw std::runtime_error("Too few arguments");

    ppf::PointCloud_t model = ppf::PointCloud_New();
    ppf::PointCloud_ReadPLY(argv[ 1 ], &model);
    ppf::PointCloud_t scene = ppf::PointCloud_New();
    ppf::PointCloud_ReadPLY(argv[ 2 ], &scene);
    ppf::PointCloud_SetViewPoint(scene, -200, -50, -500);
    {
        ppf::Timer    t("train model");
        ppf::Detector detector;
        detector.trainModel(model, 0.025f);
        detector.save("1.model");
    }

    std::vector<float> pose;
    std::vector<float> score;
    ppf::MatchResult   result;
    ppf::Detector      detector;
    detector.load("1.model");
    {
        ppf::Timer t("match scene");

        detector.matchScene(scene, pose, score, 0.025f, 0.1f, 0.1f,
                            ppf::MatchParam{55, 10, true, false, 0.5, 0, true, true, 15, 0.3},
                            &result);
    }

    // for (int i = 0; i < pose.size(); i++) {
    //     std::cout << pose[ i ] << std::endl;
    //     std::cout << score[ i ] << std::endl;
    //     auto pc = ppf::transformPointCloud(tmp, pose[ i ]);
    //     ppf::writePLY(std::string("out") + std::to_string(i) + ".ply", pc);
    // }

    ppf::PointCloud_WritePLY("sampledScene.ply", result.sampledScene);
    ppf::PointCloud_WritePLY("sampledKeypoint.ply", result.keyPoint);

    ppf::PointCloud_Delete(&model);
    ppf::PointCloud_Delete(&scene);

    return 0;
}