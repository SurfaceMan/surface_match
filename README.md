# Surface Match: fast and robust point pair feature

## highlights:
- score based multi-instance
- OpenMP & SIMD based concurrency
- build kdtree only once
- pose refined with icp
- support model save/load

## usage:
```C++
#include <ppf.h>
int main(int argc, char *argv[]) {
    ppf::PointCloud model;
    ppf::readPLY(argv[ 1 ], model);
    ppf::PointCloud scene;
    ppf::readPLY(argv[ 2 ], scene);
    scene.viewPoint = {620, 100, 500};

    {
        ppf::Detector detector;
        detector.trainModel(model, 0.04f);
        detector.save("1.model");
    }

    {
        std::vector<Eigen::Matrix4f> pose;
        std::vector<float>           score;
        ppf::Detector detector;
        detector.load("1.model");
        detector.matchScene(scene, pose, score, 0.04f, 0.1f, 0.2f);
        
        for (int i = 0; i < pose.size(); i++) {
            std::cout << pose[ i ] << std::endl;
            std::cout << score[ i ] << std::endl;
        }
    }
    return 0;
}
```