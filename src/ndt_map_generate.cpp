
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <map>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/Core>

ros::Publisher pub;

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

void
CloudCallback (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
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
    float leaf_size = 1;
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

    std::map<size_t, Leaf> leaves;
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
            std::cout << "valid" << std::endl;
        } 
        else erase_list.push_back(iter->first);
    }

    for(int erase_id=0;erase_id<erase_list.size();erase_id++){
        auto erase_iter = leaves.find(erase_list[erase_id]);
        if( erase_iter != leaves.end()) leaves.erase(erase_iter);
    }


    for(size_t i=0;i<remove_NaN_cloud->points.size();i++){//共分散合算
        int axis_v_index[3];
        axis_v_index[0] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].x * inverse_leaf_size) - static_cast<float>(min_b[0]));
        axis_v_index[1] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].y * inverse_leaf_size) - static_cast<float>(min_b[1]));
        axis_v_index[2] = static_cast<int>(std::floor(remove_NaN_cloud->points[i].z * inverse_leaf_size) - static_cast<float>(min_b[2]));
        int map_id = axis_v_index[0] * div_mul[0] + axis_v_index[1] * div_mul[1] + axis_v_index[2] * div_mul[2];

        leaves[map_id].params[0] += pow(remove_NaN_cloud->points[i].x-leaves[map_id].mean[0],2);
        leaves[map_id].params[1] += pow(remove_NaN_cloud->points[i].y-leaves[map_id].mean[1],2);
        leaves[map_id].params[2] += pow(remove_NaN_cloud->points[i].z-leaves[map_id].mean[2],2);
        leaves[map_id].params[3] += (remove_NaN_cloud->points[i].x-leaves[map_id].mean[0]) * (remove_NaN_cloud->points[i].y-leaves[map_id].mean[1]);
        leaves[map_id].params[3] += (remove_NaN_cloud->points[i].x-leaves[map_id].mean[0]) * (remove_NaN_cloud->points[i].z-leaves[map_id].mean[2]);
        leaves[map_id].params[3] += (remove_NaN_cloud->points[i].y-leaves[map_id].mean[1]) * (remove_NaN_cloud->points[i].z-leaves[map_id].mean[2]);
    }

    for(auto iter = leaves.begin(); iter != leaves.end(); ++iter){//3点以上なら平均計算それ以下なら削除
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
    }
    std::cout << "loop" << std::endl;
}

int
main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "ndt_map_generate");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe ("map_cloud", 1, CloudCallback);

    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<sensor_msgs::PointCloud2> ("normal_cloud", 1);//NDT Marker array 出す

    // Spin
    ros::spin ();
}