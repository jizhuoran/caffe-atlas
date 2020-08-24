#include <iostream>
#include <math.h>
#include "ps/ps.h"
using namespace std;
using namespace ps;


#define DEBUG

template <class Dtype>
class CaffeParamServer {
public:

    void operator() (const KVMeta& req_meta, const KVPairs<Dtype>& req_data, KVServer<Dtype>* server) {
        size_t n = req_data.keys.size();
        int work_id = (req_meta.sender - 9)/2;
#ifdef DEBUG
        std::cout << "worker id " << work_id << " " << (req_meta.push? "push" : "pull") << std::endl;
#endif


        if(req_meta.push){ //push
            if(grad.size() == 0){
                grad.resize(req_data.vals.size(), 0);
            }                
            for(int idx = 0; idx < grad.size(); ++idx){
                grad[idx] += req_data.vals[idx];
            }
            KVPairs<Dtype> res;
#ifdef DEBUG
            std::cout << "worker id: " << work_id << " response push size is " << req_data.vals.size() << std::endl;
#endif
            server->Response(req_meta, res);
        } else{ // pull
            meta_vec.push_back(req_meta);
            auto num_workers = NumWorkers();

            if(meta_vec.size() == num_workers) {
                KVPairs<Dtype> res;
                res.keys = req_data.keys;
                res.lens.resize(res.keys.size());
                res.lens[0] = grad.size();
                res.vals.resize(grad.size());
                for(int idx = 0; idx < res.lens[0]; ++idx){
                    res.vals[idx] = grad[idx] / num_workers;
                    grad[idx] = 0;
                }
                for(int work_itr = 0; work_itr < num_workers; ++work_itr) {
#ifdef DEBUG
                    std::cout << "worker id: " << work_itr << " response pull" << std::endl;
#endif
                    server->Response(meta_vec[work_itr], res);   
                }
                meta_vec.clear();
            }
        }
        
    }
private:
    std::vector<float> grad;
    std::vector<KVMeta> meta_vec;
};

void StartServer() {
    if (!IsServer()) return;
    std::cout << "num of workers[" << NumWorkers() << "]" << std::endl;
    std::cout << "num of servers[" << NumServers() << "]" << std::endl;
    auto server = new KVServer<float>(0);
    server->set_request_handle(CaffeParamServer<float>());   //注册functor
    RegisterExitCallback([server](){ delete server; });
}

int main(int argc, char* argv[]) {
    Start(0);    //启动,Postoffice::start()
    StartServer();
    Finalize(0, true); //结束。每个节点都需要执行这个函数。
    return 3;
}