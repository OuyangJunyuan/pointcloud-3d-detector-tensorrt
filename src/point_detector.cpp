//
// Created by nrsl on 23-4-3.
//
#include "inc/helper.h"
#include "inc/model.h"
#include "inc/visualization.h"

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
long time_mean = 0;
long time_counter = 0;
auto CreatMarker = [](int i, auto &&box, auto &&score, auto &&marker) {
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
};
}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "point_detector");
    auto cfg = YAML::LoadFile(canonical("config/trt.yaml").string());
    PointDetection::logger_.severity = PointDetection::str2severity[cfg["log"].as<std::string>()];

    // global lifetime detector fails to dealloc memories, use local detector instead.
    PointDetection::TRTDetector3D detector(cfg);
    cls2label = cfg["cls2label"].as<std::map<int, std::string >>();
    points.resize(detector.max_batch() * detector.max_point() * 4, 0.0f);
    detector({points.data()}); // warmup

    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<visualization_msgs::MarkerArray>("/objects", 1);
    PointDetection::MarkerArrayManager manager(pub);
    auto sub = n.subscribe<sensor_msgs::PointCloud2>("/points", 1, [&](auto &&msg) {
        ReadAndPreprocess(msg->data.data(), msg->data.size(), msg->point_step, &points);

        auto t1 = std::chrono::steady_clock::now();
        auto [boxes, scores, nums] = detector({points.data()});

        manager.Publish(CreatMarker, msg->header, nums[0][0], boxes, scores);
        auto t = time_to(t1);
        time_mean += t;
        time_counter += 1;
        printf("\033[2J"
               "=========================\n"
               "runtime: %3ld (%3ld) ms\nobjects: %3d\n"
               "=========================\n\n\n\n\n\n\n"
               "\033[0;0H",
               time_to(t1), time_mean / time_counter, int(nums[0][0]));
    });

    ros::spin();
    return 0;
}
