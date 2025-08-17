#include <bits/stdc++.h>
using namespace std;

struct Node {
    float val;                
    float grad = 0.0f;       
    vector<pair<Node*, float>> parents; 

    explicit Node(float value) : val(value) {}

    // Backward pass
    void backward(float seed = 1.0f) {
        grad += seed;
        for (auto &[parent, local_deriv] : parents) {
            parent->backward(seed * local_deriv);
        }
    }
};

using N = shared_ptr<Node>;
inline N make(float v) { return make_shared<Node>(v); }

N add(const N &a, const N &b) {
    N out = make(a->val + b->val);
    out->parents.push_back({a.get(), 1.0f});
    out->parents.push_back({b.get(), 1.0f});
    return out;
}

N sub(const N &a, const N &b) {
    N out = make(a->val - b->val);
    out->parents.push_back({a.get(), 1.0f});
    out->parents.push_back({b.get(), -1.0f});
    return out;
}

N mul(const N &a, const N &b) {
    N out = make(a->val * b->val);
    out->parents.push_back({a.get(), b->val});
    out->parents.push_back({b.get(), a->val});
    return out;
}

N divv(const N &a, const N &b) {
    if (b->val == 0.0f) { throw runtime_error("Division by zero"); }
    N out = make(a->val / b->val);
    out->parents.push_back({a.get(), 1.0f / b->val});
    out->parents.push_back({b.get(), -a->val / (b->val * b->val)});
    return out;
}

N sine(const N &a) {
    N out = make(sin(a->val));
    out->parents.push_back({a.get(), cos(a->val)});
    return out;
}

N cosine(const N &a) {
    N out = make(cos(a->val));
    out->parents.push_back({a.get(), -sin(a->val)});
    return out;
}

int main() {
    float x_val, y_val;
    cin >> x_val >> y_val;

    N x = make(x_val);
    N y = make(y_val);

    N a = divv(x, y);   // a = x / y
    N b = cosine(x);    // b = cos(x)
    N z = add(a, b);    // z = a + b

    z->backward(1.0f);

    cout << fixed << setprecision(6);
    cout << "x.val: " << x->val << " x.grad: " << x->grad << endl;
    cout << "y.val: " << y->val << " y.grad: " << y->grad << endl;
    cout << "a.val: " << a->val << " a.grad: " << a->grad << endl;
    cout << "b.val: " << b->val << " b.grad: " << b->grad << endl;
    cout << "z.val: " << z->val << " z.grad: " << z->grad << endl;

    return 0;
}
