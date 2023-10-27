1. robust icp(fast and robust icp:[paper](https://readpaper.com/paper/3042567147) [code](https://github.com/yaoyx689/Fast-Robust-ICP), robust symmetric icp:[paper](https://readpaper.com/paper/4210281455) [code](https://github.com/LJY-RS/RSICP.git))
2. ~~mesh sample~~
3. ~~serialize~~
4. c-api
5. ~~mls normal~~
6. ~~normal direction without viewpoint~~
7. edge point
8. ~~smooth point for ppf/icp? (project point on plane) mls?~~
9. serialize hash for safety
10. combine sample/normal estimation/key point?
11. point to plane icp: don't need compute scene normal, model point with normal as target in icp process
12. view based score --- most of time, only part of model have be seen in scene, that limit the match score no greater than a certain ratio. We can pre-train the visiable part in all viewpoint, then usd the closest viewpoint with target when compute score.
13. detect edge point at orgnazied point cloud(see 7.)---less point, less compute; to detect some flat things.
14. visinazation tool
15. stable normal eastimate(iter based) --- Globally Consistent Normal Orientation for Point Clouds by Regularizing the Winding-Number Field
16. port this code base to opencv 3d module
