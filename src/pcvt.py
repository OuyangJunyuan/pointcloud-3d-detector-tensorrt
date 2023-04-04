import numpy as np
from pathlib import Path
import argparse


def numpy2pcd(pts_np):
    pcd_str = ''
    pcd_str += '# .PCD v0.7 - Point Cloud Data file format\n'
    pcd_str += 'VERSION 0.7\n'
    pcd_str += 'FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
    pcd_str += 'WIDTH %d\nHEIGHT 1\n' % pts_np.shape[0]
    pcd_str += 'VIEWPOINT 0 0 0 1 0 0 0\n'
    pcd_str += 'POINTS %d\n' % pts_np.shape[0]
    pcd_str += 'DATA ascii\n'
    for pt_ in pts_np:
        pcd_str += '%f %f %f %f\n' % (pt_[0], pt_[1], pt_[2], pt_[3])
    return pcd_str


def topic2numpy(msg):
    num = msg.width * msg.height
    np_points = rnp.numpify(msg)
    fields = np_points.dtype.names
    print("msg: ", fields, num)
    pts = np.ones([num, 4])
    pts[:, 0] = np_points['x'].reshape(-1)
    pts[:, 1] = np_points['y'].reshape(-1)
    pts[:, 2] = np_points['z'].reshape(-1)
    for field in msg.fields:
        if field.name == "intensity":
            pts[:, 3] = np_points['intensity'].reshape(-1)

    return pts.astype(np.float32)


def numpy2msg(pts_np):
    msg = PointCloud2()
    msg.header.stamp = rospy.Time().now()
    msg.header.frame_id = args.frame_id

    msg.height = 1
    msg.width = len(pts_np)

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.point_step = 12

    if pts_np.shape[1] == 4:
        msg.fields.append(PointField('intensity', 12, PointField.FLOAT32, 1))
        msg.point_step = 16

    msg.is_bigendian = False
    msg.is_dense = False
    msg.row_step = msg.point_step * len(pts_np)
    msg.data = np.asarray(pts_np, np.float32).tobytes()
    return msg


def pcd2numpy(lines):
    field_, size_, type_, num_ = lines[2].split()[1:], lines[3].split()[1:], lines[4].split()[1:], lines[9].split()[1:]
    pts_ = np.zeros([int(num_[0]), len(field_)], dtype=np.float32)
    for i, line in enumerate(lines[11:]):
        pts_[i, :] = np.fromstring(line, dtype=np.float32, sep=' ')
    pts_ = np.concatenate((pts_, np.zeros_like(pts_)[:, 0][..., None]), axis=1)
    return pts_


###########################
def topic2bin(msg, save_dir):
    pts = topic2numpy(msg)

    global cnt
    cnt += 1
    save_file = save_dir / Path("%06d.bin" % cnt)

    pts.tofile(str(save_file))
    print("topic2bin save to %s" % save_file)


def topic2pcd(msg, save_dir):
    pcd_str = numpy2pcd(topic2numpy(msg))

    global cnt
    cnt += 1
    save_file = save_dir / Path("%06d.pcd" % cnt)

    with open(str(save_file), mode='w') as f:
        f.write(pcd_str)
    print("topic2pcd save to %s" % save_file)


def pcd2bin(file_path, save_dir):
    file_path, save_dir = Path(file_path), Path(save_dir)

    with open(str(file_path)) as f:
        lines = f.readlines()

    pts_ = pcd2numpy(lines)

    save_file = save_dir / Path(file_path.name).with_suffix(".bin")
    pts_.tofile(str(save_file))
    print("pcd2bin save to %s" % save_file)


def bin2pcd(file_path, save_dir):
    file_path, save_dir = Path(file_path), Path(save_dir)
    pts_ = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)

    save_file = save_dir / Path(file_path.name).with_suffix(".pcd")
    with open(str(save_file), mode='w') as f:
        f.write(numpy2pcd(pts_))
    print("bin2pcd save to %s" % save_file)


def bin2topic(file_path, p):
    file_path = Path(file_path)
    pts_ = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
    p.publish(numpy2msg(pts_))
    print("bin2topic pub to %s" % args.topic)


def pcd2topic(file_path, p):
    file_path = Path(file_path)
    with open(str(file_path)) as f:
        lines = f.readlines()

    pts_ = pcd2numpy(lines)
    p.publish(numpy2msg(pts_))
    print("pcd2topic pub to %s" % args.topic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ros msg to bin")
    parser.add_argument('-s', '--source', required=True, choices=('pcd', 'bin', 'topic'), type=str,
                        help='data source type')
    parser.add_argument('-d', '--dest', required=True, choices=('pcd', 'bin', 'topic'), type=str,
                        help='data target type')
    parser.add_argument('-p', '--path', default='./', type=str,
                        help='data path')
    parser.add_argument('-t', '--topic', default='/points', type=str,
                        help='ros topic')
    parser.add_argument('-o', '--output', default='output', type=str,
                        help='output directory')
    parser.add_argument('--frame_id', default='/map', type=str,
                        help='frame id of Point cloud when cvt to ros topic')
    args = parser.parse_args()

    assert args.source != args.dest, "no need to convert"

    path, out_path = Path(args.path), Path(args.output)
    assert path.exists()
    if not out_path.exists():
        out_path.mkdir(parents=True)

    # 读取所有文件
    cvt = globals()[args.source + '2' + args.dest]
    source_file = []
    if args.source != "topic":
        for file in path.iterdir():
            if file.suffix == '.' + args.source:
                source_file.append(file)
        source_file.sort()

        print("%d %s files totally" % (len(source_file), args.source))

    if args.source == "topic" or args.dest == "topic":
        import ros_numpy as rnp
        import rospy
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs.msg import PointField

        rospy.init_node('msg2bin_node', anonymous=True)
        cnt = 0
        if args.source == 'topic':
            rospy.Subscriber(args.topic, PointCloud2, cvt, out_path)
            rospy.spin()
        else:
            pub = rospy.Publisher(args.topic, PointCloud2, queue_size=1)
            for file in source_file:
                cvt(file, pub)
                rospy.sleep(0.05)
    else:
        for file in source_file:
            cvt(file, out_path)
