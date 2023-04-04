//
// Created by nrsl on 23-4-3.
//

#include "trt_models.hpp"

#if TRT_QUANTIZE == TRT_INT8

#include "calibrator.h"

#endif

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>


namespace {
namespace PD = PointDetection;

std::unique_ptr<PD::TRTDetector3D> detector;
pcl::PassThrough<pcl::PointXYZI>::Ptr pass_filter;
pcl::RandomSample<pcl::PointXYZI>::Ptr random_filter;
pcl::PointCloud<pcl::PointXYZI>::Ptr points;
std::vector<float> input;
int max_num_points{16384};

class MarkerArrayViz {
    ros::Publisher &publisher;
    visualization_msgs::MarkerArray markers_before;
 public:
    static auto build_marker() {
        visualization_msgs::Marker marker;

        marker.id = 0;// 同ns下同id会顶掉,不同id会共存
        marker.header.stamp = ros::Time::now();
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.2;
        marker.pose.orientation.x = 0;
        marker.pose.orientation.y = 0;
        marker.pose.orientation.z = 0;
        marker.pose.orientation.w = 1;
        marker.pose.position.x = 0;
        marker.pose.position.y = 0;
        marker.pose.position.z = 0;
        marker.color.a = 1;
        marker.color.r = 1;
        marker.lifetime = ros::Duration();
        return marker;
    }

    explicit MarkerArrayViz(ros::Publisher &pub) : publisher(pub) {}

    template<class Boxes, class F>
    void publish(Boxes &&boxes, size_t num, F &&f) {
        static visualization_msgs::MarkerArray markers_before;
        // 填充当前marker
        visualization_msgs::MarkerArray markers;
        for (int i = 0; i < num; i++) {
            visualization_msgs::Marker marker = build_marker();
            f(marker, boxes[i], i);
            markers.markers.push_back(marker);
        }
        // 添加需删去的marker
        visualization_msgs::MarkerArray marker_to_pub = markers;
        for (auto i = markers.markers.size(); i < markers_before.markers.size(); i++) {
            auto &delete_marker = markers_before.markers[i];
            delete_marker.color.a = 0.0001; // avoid warning
            marker_to_pub.markers.push_back(delete_marker);
        }
        publisher.publish(marker_to_pub);
        markers_before = markers;
    }
};

class BoundingBox {
 public:
    Eigen::Isometry3f pose{Eigen::Isometry3f::Identity()};
    Eigen::Vector3f dxyz{1, 1, 1};

    static void Corner3d(const BoundingBox &_box, Eigen::Vector3f (&_corner3d)[8]) {
        /***********************************************
         *  Box Corner  下/上
         *         2/3 __________ 6/7
         *            |    y     |
         *            |    |     |
         *            |    o - x |
         *            |          |
         *            |__________|
         *         0/1            4/5
         **********************************************/
        static Eigen::Array3f corner3d_unit[8] = {{-1, -1, -1},
                                                  {-1, -1, 1},
                                                  {-1, 1,  -1},
                                                  {-1, 1,  1},
                                                  {1,  -1, -1},
                                                  {1,  -1, 1},
                                                  {1,  1,  -1},
                                                  {1,  1,  1}};

        for (int i = 0; i < 8; i++) {
            _corner3d[i] = _box.pose * (0.5f * corner3d_unit[i] * _box.dxyz.array());
        }
    }


    static void LineList(const BoundingBox &_box, Eigen::Vector3f (&_lines)[24]) {
        static int table[] = {0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7};
        Eigen::Vector3f corners[8];
        Corner3d(_box, corners);
        for (int i = 0; i < 24; i++) {
            _lines[i][0] = corners[table[i]][0];
            _lines[i][1] = corners[table[i]][1];
            _lines[i][2] = corners[table[i]][2];
        }
    }
};

Eigen::Isometry3f inline fromXYZRPY(const Eigen::Vector3f &xyz, const Eigen::Vector3f &rpy) {
    Eigen::Isometry3f ret = Eigen::Isometry3f::Identity();
    Eigen::AngleAxisf
            r(rpy.x(), Eigen::Vector3f::UnitX()),
            p(rpy.y(), Eigen::Vector3f::UnitY()),
            y(rpy.z(), Eigen::Vector3f::UnitZ());
    ret.translate(xyz);
    ret.rotate(Eigen::Quaternionf{y * p * r});
    return ret;
}


void handler(const sensor_msgs::PointCloud2::ConstPtr &msg, ros::Publisher &pub) {
    pcl::fromROSMsg(*msg, *points);

    pass_filter->setInputCloud(points);
    pass_filter->setFilterFieldName("x");
    pass_filter->setFilterLimits(0, 70);
    pass_filter->filter(*points);
    pass_filter->setInputCloud(points);
    pass_filter->setFilterFieldName("y");
    pass_filter->setFilterLimits(-40, 40);
    pass_filter->filter(*points);
    random_filter->setInputCloud(points);
    random_filter->filter(*points);
    input.clear();
    for (int i = 0; i < points->size(); ++i) {
        input.push_back(points->points[i].x);
        input.push_back(points->points[i].y);
        input.push_back(points->points[i].z);
        input.push_back(points->points[i].intensity);
    }
    auto [boxes, scores, nums] = detector->inference(input.data());

    static MarkerArrayViz obj_viz_helper(pub);
    obj_viz_helper.publish(boxes, nums[0][0], [&msg](auto &&marker, auto &&box, int i) {
        marker.header = msg->header;
        marker.id = i;
        marker.ns = "objects";

        marker.type = visualization_msgs::Marker::LINE_LIST;
        int label = box[7];
        marker.color.r = 1;
        marker.color.g = 1;
        marker.color.b = 1;

        Eigen::Vector3f lines[24];
        BoundingBox bbox;
        bbox.pose = fromXYZRPY({box[0], box[1], box[2]}, {0, 0, box[6]});
        bbox.dxyz = {box[3], box[4], box[5]};
        BoundingBox::LineList(bbox, lines);

        geometry_msgs::Point p;
        std::for_each(std::begin(lines), std::end(lines), [&](auto &it) {
            p.x = it[0], p.y = it[1], p.z = it[2];
            marker.points.push_back(p);
        });
    });
}


}

int main(int argc, char **argv) {
    ros::init(argc, argv, "point_detector");
    ros::NodeHandle n;
    auto pub = n.advertise<visualization_msgs::MarkerArray>("/objects", 1);
    auto sub = n.subscribe<sensor_msgs::PointCloud2>("/points", 1000, boost::bind(handler, _1, boost::ref(pub)));

    auto cfg = YAML::LoadFile(PROJECT_ROOT "/config/trt.yaml");
    PD::Plugins plugins(cfg["plugins"]);
    {
        detector = std::make_unique<PD::TRTDetector3D>(cfg);
        points.reset(new pcl::PointCloud<pcl::PointXYZI>(1, max_num_points, {}));
        pass_filter.reset(new pcl::PassThrough<pcl::PointXYZI>());
        random_filter.reset(new pcl::RandomSample<pcl::PointXYZI>());
        random_filter->setSample(max_num_points);
        input.resize(max_num_points * 4, 0.0f);
    }

    ros::Rate r(1000);
    while (n.ok()) {
        ros::spinOnce();
        r.sleep();
    }

    return 0;
}

