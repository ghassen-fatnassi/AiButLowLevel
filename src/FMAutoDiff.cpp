#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define FASTIO  \
    cin.tie(0); \
    ios_base::sync_with_stdio(false);
#define TC while (t--)
const int MOD = 998244353;


class node
{
    public:
        float val;
        float diff=0;

        node(float val){
            this->val=val;
        }
        node(float val,float diff){
            this->val=val;
            this->diff=diff;
        }
        node operator*(node second_op){
            node res=node(this->val*second_op.val);
            res.diff=this->diff*second_op.val+second_op.diff*this->val;
            return res;
        }

        node operator+(node second_op){
            node res=node(this->val+second_op.val);
            res.diff=this->diff+second_op.diff;
            return res;
        }

        node operator-(node second_op){
            node res=node(this->val-second_op.val);
            res.diff=this->diff-second_op.diff;
            return res;
        }

        node operator/(node second_op){
            if(second_op.val)
            {
                node res=node(this->val / second_op.val);
                res.diff=(this->diff*second_op.val-this->val*second_op.diff)/(second_op.val*second_op.val);
                return res;
            }
            else{
                cerr << "Division by zero!\n";
                exit(1);
            }
        }

        node cosine(){
            node res=node(cos(this->val));
            res.diff=sin(this->val)*(-1)*this->diff;
            return res;
        }

        node sine(){
            node res=node(sin(this->val));
            res.diff=cos(this->val)*this->diff;
            return res;
        }
};

int main()
{
    int x_val;
    int y_val;
    cin>>x_val>>y_val;
    node x=node(x_val,1);
    node y=node(y_val,0);
    node a=x/y;
    node b=x.cosine();
    node z=a+b;
    cout <<"x.val: "<< x.val<< " x.diff :"<<x.diff<<endl;
    cout <<"y.val: "<< y.val<< " y.diff :"<<y.diff<<endl;
    cout <<"a.val: "<< a.val<< " a.diff :"<<a.diff<<endl;
    cout <<"b.val: "<< b.val<< " b.diff :"<<b.diff<<endl;
    cout <<"z.val: "<< z.val<< " z.diff :"<<z.diff<<endl;
}