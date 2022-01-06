
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <map>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

ros::Publisher vis_pub,vis_pub2;

int sort_axis=0;
int neighbor_id;
int root_id;

typedef struct
{
	int	id;
	float pos[3];
} point_with_id;

typedef struct
{
	int	parent_id;
	int left_id;
	int right_id;
	int axis;
} node;

typedef struct
{
	int	points = 0;
	// Eigen::Vector3f mean(0, 0, 0);
    float mean[3] = {0,0,0};
    float params[6]={0,0,0,0,0,0};//11,22,33,12,13,23
    float a[9];
    float eigen_vector[9];
    // Eigen::Matrix3f cov;
    // Eigen::Matrix3f evecs;
    // Eigen::Vector3f evals;
} Leaf;

std::vector<int> neighbor_list;
std::map<int, node> nodes_map;
std::map<int, Leaf> leaves;
std::vector<node> nodes;


int AxisSort(const void * n1, const void * n2)
{
	if (((point_with_id *)n1)->pos[sort_axis] > ((point_with_id *)n2)->pos[sort_axis])
	{
		return 1;
	}
	else if (((point_with_id *)n1)->pos[sort_axis] < ((point_with_id *)n2)->pos[sort_axis])
	{
		return -1;
	}
	else
	{
		return 0;
	}
}


int eigenJacobiMethod(float *a, float *v, int n, float eps = 1e-8, int iter_max = 100)
{
    float *bim, *bjm;
    float bii, bij, bjj, bji;
 
    bim = new float[n];
    bjm = new float[n];
 
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            v[i*n+j] = (i == j) ? 1.0 : 0.0;
        }
    }
 
    int cnt = 0;
    for(;;){
        int i, j;
 
        float x = 0.0;
        for(int ia = 0; ia < n; ++ia){
            for(int ja = 0; ja < n; ++ja){
                int idx = ia*n+ja;
                if(ia != ja && fabs(a[idx]) > x){
                    i = ia;
                    j = ja;
                    x = fabs(a[idx]);
                }
            }
        }
 
        float aii = a[i*n+i];
        float ajj = a[j*n+j];
        float aij = a[i*n+j];
 
        float alpha, beta;
        alpha = (aii-ajj)/2.0;
        beta  = sqrt(alpha*alpha+aij*aij);
 
        float st, ct;
        ct = sqrt((1.0+fabs(alpha)/beta)/2.0);    // sinθ
        st = (((aii-ajj) >= 0.0) ? 1.0 : -1.0)*aij/(2.0*beta*ct);    // cosθ
 
        // A = PAPの計算
        for(int m = 0; m < n; ++m){
            if(m == i || m == j) continue;
 
            float aim = a[i*n+m];
            float ajm = a[j*n+m];
 
            bim[m] =  aim*ct+ajm*st;
            bjm[m] = -aim*st+ajm*ct;
        }
 
        bii = aii*ct*ct+2.0*aij*ct*st+ajj*st*st;
        bij = 0.0;
 
        bjj = aii*st*st-2.0*aij*ct*st+ajj*ct*ct;
        bji = 0.0;
 
        for(int m = 0; m < n; ++m){
            a[i*n+m] = a[m*n+i] = bim[m];
            a[j*n+m] = a[m*n+j] = bjm[m];
        }
        a[i*n+i] = bii;
        a[i*n+j] = bij;
        a[j*n+j] = bjj;
        a[j*n+i] = bji;
 
        // V = PVの計算
        for(int m = 0; m < n; ++m){
            float vmi = v[m*n+i];
            float vmj = v[m*n+j];
 
            bim[m] =  vmi*ct+vmj*st;
            bjm[m] = -vmi*st+vmj*ct;
        }
        for(int m = 0; m < n; ++m){
            v[m*n+i] = bim[m];
            v[m*n+j] = bjm[m];
        }
 
        float e = 0.0;
        for(int ja = 0; ja < n; ++ja){
            for(int ia = 0; ia < n; ++ia){
                if(ia != ja){
                    e += fabs(a[ja*n+ia]);
                }
            }
        }
        if(e < eps) break;
 
        cnt++;
        if(cnt > iter_max) break;
    }
 
    delete [] bim;
    delete [] bjm;
 
    return cnt;
} 

int CreateNode(int* root_id,int point_size,std::vector<node>& nodes, std::vector<std::vector<int>> axis_sort_ids,int depth,int parent_id,bool node_is_right)
{
	int group_size = axis_sort_ids[0].size();
	int axis = depth % 3;
	size_t middle = ((group_size-1)/2);
	int median_id = axis_sort_ids[axis][middle];
	nodes[median_id].axis = axis;
	nodes[median_id].parent_id = parent_id;
	nodes[median_id].left_id = -1;
	nodes[median_id].right_id = -1;
	if(parent_id >= 0){ // 親あり
		if(!node_is_right) nodes[parent_id].left_id = median_id;
		if(node_is_right) nodes[parent_id].right_id = median_id;
	}
	else{ // 親なし
		*root_id = median_id;
	}

	if(group_size > 1){ // 子あり
		std::vector<int>::iterator middle_iter(axis_sort_ids[axis].begin());
		std::advance(middle_iter,middle);
		std::vector<int> left_group(axis_sort_ids[axis].begin(),middle_iter);
		++middle_iter;
		std::vector<int> right_group(middle_iter,axis_sort_ids[axis].end());
		std::cout<<std::endl;
        std::cout<<"median_id"<<median_id<<std::endl;
		std::cout<<"middle"<<middle<<std::endl;
		std::cout<<"axis"<<nodes[median_id].axis<<std::endl;
		std::cout<<"group is (";
		for(int i=0;i<group_size;i++){
			std::cout<<axis_sort_ids[axis][i]<<",";
		}
		std::cout<<")"<<std::endl;
		std::cout<<"left_group is (";
		for(int i=0;i<left_group.size();i++){
			std::cout<<left_group[i]<<",";
		}
		std::cout<<")"<<std::endl;
		std::cout<<"right_group is (";
		for(int i=0;i<right_group.size();i++){
			std::cout<<right_group[i]<<",";
		}
		std::cout<<")"<<std::endl;
        std::cout<<std::endl;
		std::vector<std::vector<int>> left_axis_sort_ids(3,std::vector<int>(left_group.size()));
		std::vector<std::vector<int>> right_axis_sort_ids(3,std::vector<int>(right_group.size()));

		std::vector<int> next_group(point_size,0);
		std::vector<int> left_axis_count(3,0);
		std::vector<int> right_axis_count(3,0);
		for(int i = 0; i < left_group.size(); i++){
			left_axis_sort_ids[axis][i] = left_group[i];
			next_group[left_group[i]] = -1;
		}
		for(int i = 0; i < right_group.size(); i++){
			right_axis_sort_ids[axis][i] = right_group[i];
			next_group[right_group[i]] = 1;
		}
		for(int i = 0; i < group_size; i++){
			for(int j = 0; j < 3; j++){
				if(j==axis) continue;
				if(next_group[axis_sort_ids[j][i]] == -1){
					left_axis_sort_ids[j][left_axis_count[j]] = axis_sort_ids[j][i];
					left_axis_count[j]++;
				}
				else if(next_group[axis_sort_ids[j][i]] == 1){
					right_axis_sort_ids[j][right_axis_count[j]] = axis_sort_ids[j][i];
					right_axis_count[j]++;
				}
			}
		}

		bool left = false;
		bool right = false;
		if(left_group.size() > 0) left = CreateNode(root_id,point_size,nodes,left_axis_sort_ids,depth+1,median_id,false);
		else left = true;

		if(right_group.size() > 0) right = CreateNode(root_id,point_size,nodes,right_axis_sort_ids,depth+1,median_id,true);
		else right = true;

		if(right&&left) return 1;
	}
	else {
        std::cout<<"leaf"<<std::endl;
        return 1;
    }
}


void
CloudCallback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    leaves.clear();
    nodes_map.clear();
    // Container for original & filtered data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointXYZ>::Ptr remove_NaN_cloud {new pcl::PointCloud<pcl::PointXYZ>};
    // Convert to PCL data type
    pcl::fromROSMsg(*cloud_msg, *cloud);
	std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloud, *remove_NaN_cloud, mapping);
    // NDT Map Generate
    // search min max point
    float min_p[3];
    float max_p[3];
    min_p[0]=remove_NaN_cloud->points[0].x;
    min_p[1]=remove_NaN_cloud->points[0].y;
    min_p[2]=remove_NaN_cloud->points[0].z;
    max_p[0]=remove_NaN_cloud->points[0].x;
    max_p[1]=remove_NaN_cloud->points[0].y;
    max_p[2]=remove_NaN_cloud->points[0].z;
    for(size_t i=0;i<remove_NaN_cloud->points.size();i++){
        if(min_p[0]>remove_NaN_cloud->points[i].x) min_p[0] = remove_NaN_cloud->points[i].x;
        if(max_p[0]<remove_NaN_cloud->points[i].x) max_p[0] = remove_NaN_cloud->points[i].x;
        if(min_p[1]>remove_NaN_cloud->points[i].y) min_p[1] = remove_NaN_cloud->points[i].y;
        if(max_p[1]<remove_NaN_cloud->points[i].y) max_p[1] = remove_NaN_cloud->points[i].y;
        if(min_p[2]>remove_NaN_cloud->points[i].z) min_p[2] = remove_NaN_cloud->points[i].z;
        if(max_p[2]<remove_NaN_cloud->points[i].z) max_p[2] = remove_NaN_cloud->points[i].z;
    }
    // calculation min max div voxel
    float leaf_size = 2;
    float inverse_leaf_size = 1 / leaf_size;
    float min_b[3];
    float max_b[3];
    float div_b[3];
    for(int i=0;i<3;i++){
        min_b[i] = static_cast<int>(std::floor(min_p[i] * inverse_leaf_size));
        max_b[i] = static_cast<int>(std::floor(max_p[i] * inverse_leaf_size));
        div_b[i] = max_b[i] - min_b[i] + 1;
    }
    float div_mul[3];
    div_mul[0] = 1;
    div_mul[1] = div_b[0];
    div_mul[2] = div_b[0] * div_b[1];

    
    for(size_t i=0;i<remove_NaN_cloud->points.size();i++){
        int axis_v_index[3];
        axis_v_index[0] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].x * inverse_leaf_size) - static_cast<float>(min_b[0]));
        axis_v_index[1] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].y * inverse_leaf_size) - static_cast<float>(min_b[1]));
        axis_v_index[2] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].z * inverse_leaf_size) - static_cast<float>(min_b[2]));
        int map_id = axis_v_index[0] * div_mul[0] + axis_v_index[1] * div_mul[1] + axis_v_index[2] * div_mul[2];

        leaves[map_id].points++;
        leaves[map_id].mean[0] += remove_NaN_cloud->points[i].x;
        leaves[map_id].mean[1] += remove_NaN_cloud->points[i].y;
        leaves[map_id].mean[2] += remove_NaN_cloud->points[i].z;
    }
    std::vector<int> erase_list; 
    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){//3点以上なら平均計算それ以下なら削除
        if(3<=iter->second.points){
            iter->second.mean[0] /= iter->second.points;
            iter->second.mean[1] /= iter->second.points;
            iter->second.mean[2] /= iter->second.points;
            // std::cout << "valid" << std::endl;
        } 
        else erase_list.push_back(iter->first);
    }

    for(int erase_id=0;erase_id<erase_list.size();erase_id++){//点削除
        auto erase_iter = leaves.find(erase_list[erase_id]);
        if( erase_iter != leaves.end()) leaves.erase(erase_iter);
    }

    for(size_t i=0;i<remove_NaN_cloud->points.size();i++){//共分散合算
        int axis_v_index[3];
        axis_v_index[0] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].x * inverse_leaf_size) - static_cast<float>(min_b[0]));
        axis_v_index[1] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].y * inverse_leaf_size) - static_cast<float>(min_b[1]));
        axis_v_index[2] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].z * inverse_leaf_size) - static_cast<float>(min_b[2]));
        int map_id = axis_v_index[0] * div_mul[0] + axis_v_index[1] * div_mul[1] + axis_v_index[2] * div_mul[2];
        auto process_iter = leaves.find(map_id);
        if (process_iter != leaves.end()){
            leaves[map_id].params[0] += pow(remove_NaN_cloud->points[i].x-leaves[map_id].mean[0],2);
            leaves[map_id].params[1] += pow(remove_NaN_cloud->points[i].y-leaves[map_id].mean[1],2);
            leaves[map_id].params[2] += pow(remove_NaN_cloud->points[i].z-leaves[map_id].mean[2],2);
            leaves[map_id].params[3] += (remove_NaN_cloud->points[i].x-leaves[map_id].mean[0]) * (remove_NaN_cloud->points[i].y-leaves[map_id].mean[1]);
            leaves[map_id].params[4] += (remove_NaN_cloud->points[i].x-leaves[map_id].mean[0]) * (remove_NaN_cloud->points[i].z-leaves[map_id].mean[2]);
            leaves[map_id].params[5] += (remove_NaN_cloud->points[i].y-leaves[map_id].mean[1]) * (remove_NaN_cloud->points[i].z-leaves[map_id].mean[2]);
        }
    }

    int max_points=0;
    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){
        if(max_points < iter->second.points){
            max_points = iter->second.points;
        }
    }

    std::cout << "max_points = " << max_points << std::endl;
    int marker_count = 0;
    visualization_msgs::MarkerArray marker_list;
    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){//3点以上なら平均計算それ以下なら削除
        if (iter == leaves.end()) continue;

        for(int param_id=0;param_id<6;param_id++){
            iter->second.params[param_id] /= iter->second.points;
        }
        iter->second.a[0] = iter->second.params[0];
        iter->second.a[1] = iter->second.params[3];
        iter->second.a[2] = iter->second.params[4];
        iter->second.a[3] = iter->second.params[3];
        iter->second.a[4] = iter->second.params[1];
        iter->second.a[5] = iter->second.params[5];
        iter->second.a[6] = iter->second.params[4];
        iter->second.a[7] = iter->second.params[5];
        iter->second.a[8] = iter->second.params[2];
        eigenJacobiMethod(iter->second.a, iter->second.eigen_vector, 3);

        if(isnan(std::sqrt(iter->second.a[0])*2)) continue;
        if(isnan(std::sqrt(iter->second.a[4])*2)) continue;
        if(isnan(std::sqrt(iter->second.a[8])*2)) continue;


        //vis
        Eigen::Matrix3f mat_rot;

        mat_rot(0,0) = iter->second.eigen_vector[0];
        mat_rot(0,1) = iter->second.eigen_vector[1];
        mat_rot(0,2) = iter->second.eigen_vector[2];
        mat_rot(1,0) = iter->second.eigen_vector[3];
        mat_rot(1,1) = iter->second.eigen_vector[4];
        mat_rot(1,2) = iter->second.eigen_vector[5];
        mat_rot(2,0) = iter->second.eigen_vector[6];
        mat_rot(2,1) = iter->second.eigen_vector[7];
        mat_rot(2,2) = iter->second.eigen_vector[8];

        Eigen::Quaternionf q_rot(mat_rot);

        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time();
        marker.ns = "my_namespace";
        marker.id = iter->first;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = iter->second.mean[0];
        marker.pose.position.y = iter->second.mean[1];
        marker.pose.position.z = iter->second.mean[2];
        marker.pose.orientation.x = q_rot.x();
        marker.pose.orientation.y = q_rot.y();
        marker.pose.orientation.z = q_rot.z();
        marker.pose.orientation.w = q_rot.w();
        marker.scale.x = std::sqrt(iter->second.a[0])*2;
        marker.scale.y = std::sqrt(iter->second.a[4])*2;
        marker.scale.z = std::sqrt(iter->second.a[8])*2;
        marker.color.a = 0.7;//iter->second.points/max_points;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        bool neighbor_voxel = false;
        // marker_list.markers.push_back(marker);
        if(marker_count<10000) {//多すぎるとバグる
            marker_list.markers.push_back(marker);
            marker_count++;
            std::cout << "mean = " << iter->second.mean[0] << "," << iter->second.mean[1] << "," << iter->second.mean[2] << std::endl;
            std::cout << "eigen_vector = ";
            for(int vis_id=0;vis_id<9;vis_id++){
                std::cout << iter->second.eigen_vector[vis_id] << " , ";
            }
            std::cout << std::endl;
            std::cout << "eigen_value = " << iter->second.a[0] << "," << iter->second.a[1] << "," << iter->second.a[2] << std::endl;
        }
    }
    // std::cout << "loop" << std::endl;
    
    
    std::cout << std::endl;
    
    std::cout << std::endl;
    std::map<size_t, Leaf> sample_leaves;
    //木を作る
	root_id=-1;
    nodes.resize(leaves.size());
	std::vector<std::vector<int>> axis_sort_ids(3,std::vector<int>(leaves.size()));
	point_with_id point_with_ids[leaves.size()];
    int point_count = 0;
    std::vector<int> index_map;
    index_map.resize(leaves.size());
	for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){//voxel
        index_map[point_count] = iter->first;
		point_with_ids[point_count].id = point_count;
		point_with_ids[point_count].pos[0] = iter->second.mean[0];//mean
		point_with_ids[point_count].pos[1] = iter->second.mean[1];
		point_with_ids[point_count].pos[2] = iter->second.mean[2];
        point_count++;
	}
	for(sort_axis=0; sort_axis<3; sort_axis++){
		qsort(point_with_ids, leaves.size(), sizeof(point_with_id), AxisSort);
		for (int i=0 ; i < leaves.size() ; i++){
			axis_sort_ids[sort_axis][i]=point_with_ids[i].id;
		}
	}
    std::cout << "sort end" << std::endl;
    std::cout << "size = " << leaves.size() << std::endl;
    std::cout << "build tree" << std::endl;
	int create_end = CreateNode(&root_id,leaves.size(),nodes,axis_sort_ids,0,-1,false);
    int root_map_id;
    std::cout << "befor root = "<< root_id << std::endl;
    root_map_id = index_map[root_id];
    root_id = root_map_id;
    std::cout << "build tree end" << std::endl;
    for(int idx=0;idx<leaves.size();idx++){//voxel
        if(0<nodes[idx].parent_id) nodes_map[index_map[idx]].parent_id = index_map[nodes[idx].parent_id];
        else nodes_map[index_map[idx]].parent_id = -1;
        if(0<nodes[idx].left_id) nodes_map[index_map[idx]].left_id = index_map[nodes[idx].left_id];
        else nodes_map[index_map[idx]].left_id = -1;
        if(0<nodes[idx].right_id) nodes_map[index_map[idx]].right_id = index_map[nodes[idx].right_id];
        else nodes_map[index_map[idx]].right_id = -1;
        nodes_map[index_map[idx]].axis = nodes[idx].axis;
	}
    vis_pub.publish(marker_list);
}

double EuclidDist3D(const float* a, const float* b){
    float d2 = 0;
    for(int i = 0; i < 3; ++i){
        d2 += std::pow(a[i] - b[i], 2);
    }
    return std::sqrt(d2);
}

void searchRecursive(const float* query_position,
                    const std::map <int, Leaf> &leaves,
                    const std::map <int, node> &tree,
                    const int &node_id,
                    double &min_dist){
    // reach leave.
    if(node_id < 0){
        return;
    }
    // std::cout << " node_id = " << node_id;
    auto tree_iter = tree.find(node_id);
    auto leaf_iter = leaves.find(node_id);

    node n = tree_iter->second;
    Leaf l = leaf_iter->second;
    // std::cout << "dist" << std::endl;
    double dist = EuclidDist3D(l.mean, query_position);

    if(dist < min_dist){
        min_dist = dist;
        neighbor_id = node_id;
    }

    int next_id;
    int opp_id;
    // std::cout << "left&right" << std::endl;
    if(query_position[n.axis] < l.mean[n.axis]){
        next_id = n.left_id;///ここ見る
        opp_id = n.right_id;
    }else{
        next_id = n.right_id;///ここ見る
        opp_id = n.left_id;
    }

    searchRecursive(query_position, leaves, tree, next_id, min_dist);

    double diff = std::fabs(query_position[n.axis] - l.mean[n.axis]);
    if (diff < min_dist)
        searchRecursive(query_position, leaves, tree, opp_id, min_dist);
}

void rangeSearchRecursive(const float* query_position,
                     const std::map <int, Leaf> &leaves,
                     const std::map <int, node> &tree,
                     const int &node_id,
                     double &search_range){
    // reach leave.
    if(node_id < 0){
        return;
    }

    node n = tree.at(node_id);
    Leaf l = leaves.at(node_id);

    double dist = EuclidDist3D(l.mean, query_position);

    if(dist < search_range){
        neighbor_list.push_back(node_id);
    }

    int next_id;
    int opp_id;
    if(query_position[n.axis] < l.mean[n.axis]){
        next_id = n.left_id;
        opp_id = n.right_id;
    }else{
        next_id = n.right_id;
        opp_id = n.left_id;
    }

    rangeSearchRecursive(query_position, leaves, tree, next_id, search_range);
    double diff = std::fabs(query_position[n.axis] - l.mean[n.axis]);
    if (diff < search_range) rangeSearchRecursive(query_position, leaves, tree, opp_id, search_range);
}

void
PoseCallback (const geometry_msgs::PoseStampedConstPtr & pose_msg)
{
    std::cout << "Pose callback" << std::endl;
    //座標取得
    float target[3];
    target[0] = pose_msg->pose.position.x;
    target[1] = pose_msg->pose.position.y;
    target[2] = pose_msg->pose.position.z;

    

    //近傍探索
    std::cout << "search start" << std::endl;
    std::cout << "root_id = " << root_id << std::endl;
    // float root_mean[3];
    // root_mean[0]=leaves[root_id].mean[0];
    // root_mean[1]=leaves[root_id].mean[1];
    // root_mean[2]=leaves[root_id].mean[2];
    // double min_dist=EuclidDist3D(root_mean, target);
    // searchRecursive(target,leaves,nodes_map,root_id,min_dist);
    // std::cout<< "neighbor_id = "<<neighbor_id<<std::endl;

    double search_range = 10;
    rangeSearchRecursive(target,leaves,nodes_map,root_id,search_range);
    visualization_msgs::MarkerArray marker_list2;
    for(int neighbor_count=0;neighbor_count<neighbor_list.size();neighbor_count++){
        std::cout<<"neighbor_id = "<< neighbor_list[neighbor_count]<<std::endl;
        auto vis_iter = leaves.find(neighbor_list[neighbor_count]);

        if(isnan(std::sqrt(vis_iter->second.a[0])*2)) continue;
        if(isnan(std::sqrt(vis_iter->second.a[4])*2)) continue;
        if(isnan(std::sqrt(vis_iter->second.a[8])*2)) continue;
        Eigen::Matrix3f mat_rot2;
        mat_rot2(0,0) = vis_iter->second.eigen_vector[0];
        mat_rot2(0,1) = vis_iter->second.eigen_vector[1];
        mat_rot2(0,2) = vis_iter->second.eigen_vector[2];
        mat_rot2(1,0) = vis_iter->second.eigen_vector[3];
        mat_rot2(1,1) = vis_iter->second.eigen_vector[4];
        mat_rot2(1,2) = vis_iter->second.eigen_vector[5];
        mat_rot2(2,0) = vis_iter->second.eigen_vector[6];
        mat_rot2(2,1) = vis_iter->second.eigen_vector[7];
        mat_rot2(2,2) = vis_iter->second.eigen_vector[8];

        Eigen::Quaternionf q_rot2(mat_rot2);

        visualization_msgs::Marker marker2;
        marker2.header.frame_id = "map";
        marker2.header.stamp = ros::Time();
        marker2.ns = "my_namespace";
        marker2.id = vis_iter->first;
        marker2.type = visualization_msgs::Marker::SPHERE;
        marker2.action = visualization_msgs::Marker::ADD;
        marker2.pose.position.x = vis_iter->second.mean[0];
        marker2.pose.position.y = vis_iter->second.mean[1];
        marker2.pose.position.z = vis_iter->second.mean[2];
        marker2.pose.orientation.x = q_rot2.x();
        marker2.pose.orientation.y = q_rot2.y();
        marker2.pose.orientation.z = q_rot2.z();
        marker2.pose.orientation.w = q_rot2.w();
        marker2.scale.x = std::sqrt(vis_iter->second.a[0])*2*1.5;
        marker2.scale.y = std::sqrt(vis_iter->second.a[4])*2*1.5;
        marker2.scale.z = std::sqrt(vis_iter->second.a[8])*2*1.5;
        marker2.color.a = 1.0;//vis_iter->second.points/max_points;
        marker2.color.r = 1.0;
        marker2.color.g = 0.0;
        marker2.color.b = 0.0;
        marker_list2.markers.push_back(marker2);
    }

    vis_pub2.publish(marker_list2);
}

int
main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "ndt_map_generate");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber map_sub = nh.subscribe ("map_cloud", 1, CloudCallback);
    ros::Subscriber pose_sub = nh.subscribe ("move_base_simple/goal", 1, PoseCallback);

    // Create a ROS publisher for the output point cloud
    vis_pub = nh.advertise<visualization_msgs::MarkerArray>("/ndt_ellipsoid", 10);
    vis_pub2 = nh.advertise<visualization_msgs::MarkerArray>("/neighbor_ellipsoid", 10);
    // Spin
    ros::spin ();
}