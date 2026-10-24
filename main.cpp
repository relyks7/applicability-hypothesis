#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>
#include <random>
random_device rd;
mt19937 rng(rd());
using namespace std;
#define all(v) v.begin(), v.end()
void matvec(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m){
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, 1.0f, a.data(), m, b.data(), 1, 0.0f, c.data(), 1);
}
void matvec_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m){
    cblas_sgemv(CblasRowMajor, CblasTrans, n, m, 1.0f, a.data(), m, b.data(), 1, 0.0f, c.data(), 1);
}
void matmat(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int p){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, p, m, 1.0f, a.data(), m, b.data(), p, 0.0f, c.data(), p);
}
void matmat_transpose(const vector<float>& a, const vector<float>& b, vector<float>& c, int n, int m, int p){
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, p, m, 1.0f, a.data(), m, b.data(), m, 0.0f, c.data(), p);
}
float gaussian_noise(float mean, float stddev){
    normal_distribution<float> dist(mean, stddev);
    return dist(rng);
}
float uniform(float lo, float hi){
    uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}
float sgmd(float x){
    return 1.0f/(1.0f+exp(-x));
}
float eps=0.003f;
class l_p{ //perceptron layer, tanh/sigmoid
    public:
    vector<float> w;
    vector<float> b;
    vector<float> ipt;
    vector<float> opt;
    vector<float> e;
    int d_in, d_out;
    int tp;
    float lr_w, lr_b;
    void r(){
        w.assign(d_in*d_out, 0.0f); for (int i=0;i<d_in*d_out;i++) w[i]=gaussian_noise(0.0f, 1.0f/sqrtf(d_in));
        b.assign(d_out, 0.0f);
        ipt.assign(d_in, 0.0f);
        opt.assign(d_out, 0.0f);
        e.assign(d_in, 0.0f);
    }
    l_p(int din, int dout, float lrw, float lrb, int ftp){
        d_in=din; d_out=dout; lr_w=lrw; lr_b=lrb; tp=ftp;
        r();
    }
    vector<float> fp(vector<float>x){
        ipt=x;
        matvec(w, ipt, opt, d_out, d_in);
        for (int i=0;i<d_out;i++) opt[i]=tp?sgmd(opt[i]+b[i]):tanh(opt[i]+b[i]); 
        return opt;
    }
    vector<float> bp(vector<float>& g){
        matvec_transpose(w, g, e, d_out, d_in);
        for (int i=0;i<d_out;i++) g[i]*=tp?(opt[i]*(1.0f-opt[i])):(1.0f-opt[i]*opt[i]);
        for (int i=0;i<d_out;i++) b[i]-=lr_b*g[i];
        for (int i=0;i<d_out;i++) for (int j=0;j<d_in;j++) w[i*d_in+j]-=lr_w*g[i]*ipt[j];
        return e;
    }
};
class h{ //hypothesis, mlp
    public:
    float lr_w, lr_b;
    int d_in;
    vector<l_p> l{};
    h(int din, float lrw, float lrb){
        d_in=din; lr_w=lrw; lr_b=lrb;
        l.push_back(l_p(d_in, 32, lr_w, lr_b, 0));
        l.push_back(l_p(32, 32, lr_w, lr_b, 0));
        l.push_back(l_p(32, 1, lr_w, lr_b, 1));
    }
    vector<float> f(vector<float>x){
        for (int i=0;i<l.size();i++) x=l[i].fp(x);
        return x;
    }
    void b(vector<float>& e){
        for (int i=l.size()-1;i>=0;i--) e=l[i].bp(e);
    }
};
class K{ //world knowledge
    public:
    int d;
    float lr_w, lr_b;
    float lo=-3.0f, hi=3.0f;
    int tks=0;
    vector<h> m{};
    vector<vector<float>> rs{};
    vector<vector<float>> cs{};
    h cur{0, 0.0f, 0.0f};
    K(int din, float lrw, float lrb){
        d=din; lr_w=lrw; lr_b=lrb;
        cur=h(din, lrw, lrb);
    }
    void t_r(vector<float>& r){ //train on reality
        float p=cur.f(r)[0];
        vector<float> e{p};
        rs.push_back(r);
        cur.b(e);
    }
    float s(vector<float> c){ //score
        float ret=0.0f;
        for (auto th:m){
            ret+=th.f(c)[0]*th.f(c)[0];
        }
        return ret;
    }
    int t_c(){ //anneal to find counterfactual, then train on it
        vector<float> c(d, 0.0f);
        vector<float> nc(d, 0.0f);
        for (int i=0;i<d;i++) c[i]=uniform(lo, hi);
        for (int i=0;i<100000;i++){
            nc=c;
            for (int j=0;j<d;j++) nc[j]+=gaussian_noise(0.0f, 0.05f);
            if (s(nc)<s(c)) swap(c, nc);
            if (s(c)<eps) break;
        }
        if (s(c)>eps) return 0;
        float p=cur.f(c)[0];
        vector<float> e{p-1.0f};
        cur.b(e);
        cs.push_back(c);
        return 1;
    }
    void rst(){
        tks=0;
        cur=h(d, lr_w, lr_b);
        rs.clear();
        cs.clear();
    }
    bool fit(){
        float s_r=0.0f;
        for (auto r:rs){
            float p=cur.f(r)[0];
            s_r+=0.5f*p*p;
        }
        float s_c=0.0f;
        for (auto r:cs){
            float p=cur.f(r)[0];
            s_c+=0.5f*(p-1.0f)*(p-1.0f);
        }
        return s_r+s_c<0.05f; //refine this later
    }
    void train(vector<float> r){
        if (tks>=100000){
            if (fit()) m.push_back(cur);
            rst();
        }
        t_r(r);
        t_c();
        tks++;
    }
};
int main(){
    
    return 0;
}