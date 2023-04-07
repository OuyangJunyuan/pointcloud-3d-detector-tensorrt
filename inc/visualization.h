//
// Created by nrsl on 23-4-5.
//

#ifndef POINT_DETECTION_VISUALIZATION_H
#define POINT_DETECTION_VISUALIZATION_H

#include <Eigen/Geometry>
#include <ros/publisher.h>
#include <visualization_msgs/MarkerArray.h>

namespace PointDetection {
struct BoundingBox {
    BoundingBox() : pose{Eigen::Isometry3f::Identity()}, dxyz({1, 1, 1}) {}

    BoundingBox(float x, float y, float z, float dx, float dy, float dz, float rx, float ry, float rz) {
        pose = Eigen::Isometry3f::Identity();
        Eigen::AngleAxisf
                arx(rx, Eigen::Vector3f::UnitX()),
                ary(ry, Eigen::Vector3f::UnitY()),
                arz(rz, Eigen::Vector3f::UnitZ());
        pose.translate(Eigen::Vector3f{x, y, z});
        pose.rotate(Eigen::Quaternionf{arz * ary * arx});
        dxyz = {dx, dy, dz};
    }

    static void GetCorner3d(const BoundingBox &box, Eigen::Vector3f (&corner3d)[8]) {
        /***********************************************
         *  Box Corner(upper/lower)
         *         2/3 __________ 6/7
         *            |    y     |
         *            |    |     |
         *            |    o - x |
         *            |          |
         *            |__________|
         *         0/1            4/5
         **********************************************/
        static const Eigen::Array3f kCorner3dUnit[8] = {{-1, -1, -1},
                                                        {-1, -1, 1},
                                                        {-1, 1,  -1},
                                                        {-1, 1,  1},
                                                        {1,  -1, -1},
                                                        {1,  -1, 1},
                                                        {1,  1,  -1},
                                                        {1,  1,  1}};

        for (int i = 0; i < 8; i++) {
            corner3d[i] = box.pose * (0.5f * kCorner3dUnit[i] * box.dxyz.array());
        }
    }

    static void GetLineList(const BoundingBox &box, Eigen::Vector3f (&lines)[24]) {
        static constexpr int kCorner3dOrders[] = {0, 1,
                                                  1, 3,
                                                  3, 2,
                                                  2, 0,
                                                  4, 5,
                                                  5, 7,
                                                  7, 6,
                                                  6, 4,
                                                  0, 4,
                                                  1, 5,
                                                  2, 6,
                                                  3, 7};
        Eigen::Vector3f corners[8];
        GetCorner3d(box, corners);
        for (int i = 0; i < 24; i++) {
            lines[i] = corners[kCorner3dOrders[i]];
        }
    }

    static void GetLineList(const BoundingBox &_box, Eigen::Vector3f (&lines)[28]) {
        static constexpr int kCorner3dOrders[] = {0, 1,
                                                  1, 3,
                                                  3, 2,
                                                  2, 0,
                                                  4, 5,
                                                  5, 7,
                                                  7, 6,
                                                  6, 4,
                                                  0, 4,
                                                  1, 5,
                                                  2, 6,
                                                  3, 7,
                                                  6, 5,
                                                  4, 7};
        Eigen::Vector3f corners[8];
        GetCorner3d(_box, corners);
        for (int i = 0; i < 28; i++) {
            lines[i] = corners[kCorner3dOrders[i]];
        }
    }

    static void GetLineList(float x, float y, float z, float dx, float dy, float dz, float rx, float ry, float rz,
                            Eigen::Vector3f (&lines)[28]) {
        GetLineList(BoundingBox(x, y, z, dx, dy, dz, rx, ry, rz), lines);
    }

    static void GetLineList(float x, float y, float z, float dx, float dy, float dz, float rx, float ry, float rz,
                            Eigen::Vector3f (&lines)[24]) {
        GetLineList(BoundingBox(x, y, z, dx, dy, dz, rx, ry, rz), lines);
    }


    Eigen::Isometry3f pose;
    Eigen::Vector3f dxyz;
};

class MarkerArrayManager {
 public:
    explicit MarkerArrayManager(const ros::Publisher &pub)
            : publisher_(pub) {
    }

    auto BuildDefaultMarker() {
        visualization_msgs::Marker marker;

        marker.id = 0;  // objects must have different id in the same ns.
        marker.header = header_;
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


    template<class ...T, class F>
    void Publish(F &&handler, const std_msgs::Header &header, size_t num, T &&...args) {
        header_ = header;
        visualization_msgs::MarkerArray marker_to_pub;
        visualization_msgs::Marker delete_marker = BuildDefaultMarker();
        delete_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_to_pub.markers.push_back(delete_marker);

        for (int i = 0; i < num; i++) {
            visualization_msgs::Marker marker = BuildDefaultMarker();
            handler(i, std::forward<T>(args)[i]..., marker);
            marker_to_pub.markers.push_back(marker);
        }
        publisher_.publish(marker_to_pub);
    }

 private:
    const ros::Publisher &publisher_;
    std_msgs::Header header_;
};

}  // namespace PointDetection

#endif //POINT_DETECTION_VISUALIZATION_H
