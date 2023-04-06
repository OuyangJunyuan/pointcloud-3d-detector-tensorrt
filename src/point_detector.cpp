//
// Created by nrsl on 23-4-3.
//
#include "inc/tensorrt_model.h"
#include "inc/visualization_helper.h"

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>

namespace {
std::map<int, std::string> cls2label;

std::vector<float> points;
std::unique_ptr<PointDetection::TRTDetector3D> detector;

void ReadMsgAndPreprocess(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    points.reserve(msg->height * msg->width * 4);
    auto ptr_cur = msg->data.data();
    auto ptr_end = ptr_cur + msg->data.size();
    auto ptr_tar = points.data();
    for (; ptr_cur <= ptr_end; ptr_cur += msg->point_step) {
        auto *p = reinterpret_cast<const float *>(ptr_cur);
        auto x = p[0], y = p[1], z = p[2], intensity = p[3];
        if ((0 < x and x < 70) and (-45 < y and y < 45) and 16 < x * x + y * y + z * z) {
            ptr_tar[0] = x, ptr_tar[1] = y, ptr_tar[2] = z, ptr_tar[3] = intensity;
            ptr_tar += 4;
        }
    }
    std::random_shuffle(reinterpret_cast<typeof(float[4]) *>((void *) points.data()),
                        reinterpret_cast<typeof(float[4]) *>(ptr_tar));
}

void Handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const ros::Publisher &pub) {
    std::cout << "=========================" << std::endl;
    ReadMsgAndPreprocess(msg);

    auto t1 = std::chrono::steady_clock::now();
    auto [boxes, scores, nums] = detector->Infer({points.data()});
    auto t2 = std::chrono::steady_clock::now();

    static PointDetection::MarkerArrayManager manager(pub, msg->header);
    manager.Publish([](int i, auto &&box, auto &&score, auto &&marker) {
        const auto &[x, y, z, dx, dy, dz, heading, cls] = box;

        marker.id = i;
        marker.ns = cls2label[static_cast<int>(cls)];
        marker.type = visualization_msgs::Marker::LINE_LIST;

        marker.color.r = 1;
        marker.color.g = 1;
        marker.color.b = 1;

        Eigen::Vector3f lines[28];
        PointDetection::BoundingBox::GetLineList(x, y, z, dx, dy, dz, 0, 0, heading, lines);

        geometry_msgs::Point p;
        for (auto &&l: lines) {
            p.x = l[0], p.y = l[1], p.z = l[2];
            marker.points.push_back(p);
        }
    }, nums[0][0], boxes, scores);

    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("runtime: %ld\n"
           "objects: %d\n",
           runtime, int(nums[0][0]));
}

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "point_detector");

    ros::NodeHandle n;
    auto pub = n.advertise<visualization_msgs::MarkerArray>("/objects", 1);
    auto sub = n.subscribe<sensor_msgs::PointCloud2>("/points", 100, boost::bind(Handler, _1, boost::ref(pub)));

    auto cfg = YAML::LoadFile(canonical("config/trt.yaml").string());
    cls2label = cfg["cls2label"].as<std::map<int, std::string>>();
    detector = std::make_unique<PointDetection::TRTDetector3D>(cfg);

    points.resize(detector->max_batch_size_ * detector->max_point_num_ * 4, 0.0f);

    ros::Rate r(1000);
    while (n.ok()) {
        ros::spinOnce();
        r.sleep();
    }

    return 0;
}
