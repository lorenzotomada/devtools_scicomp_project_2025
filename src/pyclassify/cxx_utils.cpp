#include <iostream>
#include <utility>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <functional>
#include <tuple>
#include <numeric>
#include <stdexcept>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <pybind11/numpy.h>


namespace py=pybind11;	
#include <pybind11/eigen.h>


std::pair<std::vector<double>, std::vector<std::vector<double>> > 
QR_algorithm(std::vector<double>  diag, std::vector<double>  off_diag, const double tol=1e-8, const unsigned int max_iter=5000){

    if(diag.size() != (off_diag.size()+1)){
        std::invalid_argument("The dimension of the diagonal and off-diagonal vector are not compatible");
    }

    const unsigned int n = diag.size();

    //std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0)), R(n);
    std::vector<double> Q(n*n);

    for(unsigned int i = 0; i < n*n; i=i+n+1){
        Q[i] = 1;
    }


    std::vector<std::array<double, 2>> Matrix_trigonometric(n-1, {0, 0});

    unsigned int iter = 0;  
    std::vector<double> eigenvalues_old{diag};

    double r=0, c=0, s=0;
    double d=0, mu; // mu: Wilkinson shift
    double a_m=0, b_m_1=0;
    double tmp=0;
    double x=0, y=0;
    unsigned int m=n-1;
    double tol_equivalence=1e-10;
    double w=0, z=0;

    

    while (iter<max_iter && m>0)
    {
        // prefetching most used value to avoid call overhead
        a_m=diag[m];
        b_m_1=off_diag[m-1];
        d=(diag[m-1]-a_m)*0.5;
        
        if(std::abs(d)<tol_equivalence){
            mu=diag[m]-std::abs(b_m_1);
        } else{
            mu= a_m - b_m_1*b_m_1/( d*( 1+sqrt(d*d+b_m_1*b_m_1)/std::abs(d)  ) );
        }

        x=diag[0]-mu;
        y=off_diag[0];

        for(unsigned int i=0; i<m; i++){
            if (m>1)
            {
                r=std::sqrt(x*x+y*y);
                c=x/r;
                s=-y/r;
                Matrix_trigonometric[i][0] = c;
                Matrix_trigonometric[i][1] = s;
                
                w=c*x-s*y;
                d=diag[i]-diag[i+1];
                z=(2*c*off_diag[i] +d*s)*s;
                diag[i] -= z;
                diag[i+1] += z;
                off_diag[i]= d*c*s + (c*c-s*s)*off_diag[i];
                x=off_diag[i];
                if (i>0)
                {
                    off_diag[i-1]=w;
                }
    
                if(i<m-1){
                    y=-s*off_diag[i+1];
                    off_diag[i+1]=c*off_diag[i+1];
                }
    
            }else{
                if(std::abs(d)<tol_equivalence){
                    if (off_diag[0]*d>0)
                    {
                        c=std::sqrt(2)/2;
                        s=-std::sqrt(2)/2;
                    } else{
                        c=s=std::sqrt(2)/2;
                    }
                    
                } else{
                    double x_0=0, x_new=0, b_2=off_diag[0];
                    if(off_diag[0]*d>0){
                        x_0=-3.14/4;
                    } else{
                        x_0=3.14/4;
                    }
                    double err_rel=1;
                    unsigned int iter_newton=0;
                    while (err_rel>1e-10 && iter_newton<1000)
                    {
                        x_new=x_0-std::cos(x_0)*std::cos(x_0)*(std::tan(x_0) + b_2/d);
                        err_rel=std::abs((x_new-x_0)/x_new);
                        x_0=x_new;
                        ++iter_newton;
                    }
                    c=std::cos(x_new/2);
                    s=std::sin(x_new/2);

                    Matrix_trigonometric[i][0] = c;
                    Matrix_trigonometric[i][1] = s;  
                    
                    double a_0=diag[0], b_1=off_diag[0];

                    off_diag[0]=0; //c*s*(a_0-diag[1])+b_1*(c*c-s*s);
                    diag[0]=c*c*a_0+s*s*diag[1]-2*s*c*b_1;
                    diag[1]=c*c*diag[1]+s*s*a_0+2*s*c*b_1;
                    

                }

            }

      
        }



        unsigned j, k;
        for(unsigned int i=0; i<m; i++){
            c=Matrix_trigonometric[i][0];
            s=Matrix_trigonometric[i][1];
            
            for(j=0; (j+5)<n;j=j+5){
                k=n*i+j;
                tmp=Q[k];
                Q[k]=tmp*c-Q[k+n]*s;
                Q[k+n]=tmp*s+Q[k+n]*c;
                tmp=Q[k+1];
                Q[k+1]=tmp*c-Q[k+n+1]*s;
                Q[k+n+1]=tmp*s+Q[k+n+1]*c;
                tmp=Q[k+2];
                Q[k+2]=tmp*c-Q[k+n+2]*s;
                Q[k+n+2]=tmp*s+Q[k+n+2]*c;
                tmp=Q[k+3];
                Q[k+3]=tmp*c-Q[k+n+3]*s;
                Q[k+n+3]=tmp*s+Q[k+n+3]*c;               
                tmp=Q[k+4];
                Q[k+4]=tmp*c-Q[k+n+4]*s;
                Q[k+n+4]=tmp*s+Q[k+n+4]*c;
            }
            for(; j < n; j++)
            {
                k=n*i+j;
                tmp=Q[k];
                Q[k]=tmp*c-Q[k+n]*s;
                Q[k+n]=tmp*s+Q[k+n]*c;
            }
    
            
        }

        iter++;
        if ( std::abs(off_diag[m-1]) < tol*( std::abs(diag[m]) + std::abs(diag[m-1]) )  )
        {
            --m;
        }
    }

    if(iter==max_iter){
        std::cout<<"Converges failed"<<std::endl;
    }

    std::vector<std::vector<double>> eig_vec(n,std::vector<double> (n, 0));
    //std::cout<<"Iteration: "<<iter<<std::endl;
    
    for(unsigned j=0; j<n; j++){
        for(unsigned int i=0; i<n; i++){
            eig_vec[i][j]=Q[i+j*n];
        }
    }

    return std::make_pair(diag, eig_vec);
 

}

std::vector<double>
Eigen_value_calculator(std::vector<double> diag, std::vector<double> off_diag, const double tol=1e-8, const unsigned int max_iter=5000){

    if(diag.size() != (off_diag.size()+1)){
        std::invalid_argument("The dimension of the diagonal and off-diagonal vector are not compatible");
    }

    const unsigned int n = diag.size();




    std::vector<std::array<double, 2>> Matrix_trigonometric(n-1, {0, 0});

    unsigned int iter = 0;  
    std::vector<double> eigenvalues_old{diag};

    double r=0, c=0, s=0;
    double d=0, mu; // mu: Wilkinson shift
    double a_m=0, b_m_1=0;
    double x=0, y=0;
    unsigned int m=n-1;
    double tol_equivalence=1e-10;
    double w=0, z=0;

    

    while (iter<max_iter && m>0)
    {
        // prefetching most used value to avoid call overhead
        a_m=diag[m];
        b_m_1=off_diag[m-1];
        d=(diag[m-1]-a_m)*0.5;
        
        if(std::abs(d)<tol_equivalence){
            mu=diag[m]-std::abs(b_m_1);
        } else{
            mu= a_m - b_m_1*b_m_1/( d*( 1+sqrt(d*d+b_m_1*b_m_1)/std::abs(d)  ) );
        }

        x=diag[0]-mu;
        y=off_diag[0];

        for(unsigned int i=0; i<m; i++){
            if (m>1)
            {
                r=std::sqrt(x*x+y*y);
                c=x/r;
                s=-y/r;
                Matrix_trigonometric[i][0] = c;
                Matrix_trigonometric[i][1] = s;
                
                w=c*x-s*y;
                d=diag[i]-diag[i+1];
                z=(2*c*off_diag[i] +d*s)*s;
                diag[i] -= z;
                diag[i+1] += z;
                off_diag[i]= d*c*s + (c*c-s*s)*off_diag[i];
                x=off_diag[i];
                if (i>0)
                {
                    off_diag[i-1]=w;
                }
    
                if(i<m-1){
                    y=-s*off_diag[i+1];
                    off_diag[i+1]=c*off_diag[i+1];
                }
    
            }else{
                if(std::abs(d)<tol_equivalence){
                    if (off_diag[0]*d>0)
                    {
                        c=std::sqrt(2)/2;
                        s=-std::sqrt(2)/2;
                    } else{
                        c=s=std::sqrt(2)/2;
                    }
                    
                } else{
                    double x_0=0, x_new=0, b_2=off_diag[0];
                    if(off_diag[0]*d>0){
                        x_0=-3.14/4;
                    } else{
                        x_0=3.14/4;
                    }
                    double err_rel=1;
                    unsigned int iter_newton=0;
                    while (err_rel>1e-10 && iter_newton<1000)
                    {
                        x_new=x_0-std::cos(x_0)*std::cos(x_0)*(std::tan(x_0) + b_2/d);
                        err_rel=std::abs((x_new-x_0)/x_new);
                        x_0=x_new;
                        ++iter_newton;
                    }
                    c=std::cos(x_new/2);
                    s=std::sin(x_new/2);

                    Matrix_trigonometric[i][0] = c;
                    Matrix_trigonometric[i][1] = s;  
                    
                    double a_0=diag[0], b_1=off_diag[0];

                    off_diag[0]=0; //c*s*(a_0-diag[1])+b_1*(c*c-s*s);
                    diag[0]=c*c*a_0+s*s*diag[1]-2*s*c*b_1;
                    diag[1]=c*c*diag[1]+s*s*a_0+2*s*c*b_1;
                    

                }

            }

      
        }


        iter++;
        if ( std::abs(off_diag[m-1]) < tol*( std::abs(diag[m]) + std::abs(diag[m-1]) )  )
        {
            --m;
        }
    }

    if(iter==max_iter){
        std::cout<<"The method did not converge"<<std::endl;
    }


    return diag;
}


double compute_sum(
    const std::vector<double>& v,
    const std::vector<double>& d,
    const double x,
    const unsigned int start,
    const unsigned int end,
    const bool derivative,
    const double rho) {

    double sum{0.0};
    for (unsigned int j = start; j < end; ++j) {
        double denom = d[j] - x;
        if (derivative) {
            sum += v[j] * v[j] / (denom * denom);
        } else {
            sum += v[j] * v[j] / denom;
        }
    }
    return rho * sum;
}

void compute_Psi(
    const unsigned int i,
    const std::vector<double>& v,
    const std::vector<double>& d,
    const double rho,
    std::function<double(double)>& Psi_1,
    std::function<double(double)>& Psi_2,
    std::function<double(double)>& dPsi_1,
    std::function<double(double)>& dPsi_2
) {
    Psi_1 = [&](double x) {
        return compute_sum(v, d, x, 0, i + 1, false, rho);
    };
    Psi_2 = [&](double x) {
        return compute_sum(v, d, x, i + 1, static_cast<int>(d.size()), false, rho);
    };
    dPsi_1 = [&](double x) {
        return compute_sum(v, d, x, 0, i + 1, true, rho);
    };
    dPsi_2 = [&](double x) {
        return compute_sum(v, d, x, i + 1, static_cast<int>(d.size()), true, rho);
    };
}

std::pair<double, double> find_root(
    const unsigned int i,
    const bool left_center,
    const std::vector<double>& v,
    const std::vector<double>& d,
    const double rho,
    double lam_0,
    const double tol = 1e-15,
    const unsigned int maxiter = 100) {
    std::vector<double> diag = d;
    double shift;

    if (left_center) {
        for (size_t j = 0; j < diag.size(); ++j) {
            diag[j] -= d[i];
        }
        lam_0 -= d[i];
        shift = d[i];
    } else {
        for (size_t j = 0; j < diag.size(); ++j) {
            diag[j] -= d[i + 1];
        }
        lam_0 -= d[i + 1];
        shift = d[i + 1];
    }

    std::function<double(double)> Psi_1, Psi_2, dPsi_1, dPsi_2;
    compute_Psi(i, v, diag, rho, Psi_1, Psi_2, dPsi_1, dPsi_2);

    for (unsigned int iter = 0; iter < maxiter; ++iter) {
        double delta_i = diag[i] - lam_0;
        double delta_i1 = (i + 1 < static_cast<unsigned int>(d.size())) ? (diag[i + 1] - lam_0) : 0.0;
        double vPsi_1 = Psi_1(lam_0);
        double vPsi_2 = Psi_2(lam_0);
        double vdPsi_1 = dPsi_1(lam_0);
        double vdPsi_2 = dPsi_2(lam_0);

        double a = (1.0 + vPsi_1 + vPsi_2) * (delta_i + delta_i1)
                 - (vdPsi_1 + vdPsi_2) * delta_i * delta_i1;
        double b = delta_i * delta_i1 * (1.0 + vPsi_1 + vPsi_2);
        double c = 1.0 + vPsi_1 + vPsi_2 - delta_i * vdPsi_1 - delta_i1 * vdPsi_2;
        double discr = a * a - 4.0 * b * c;
        discr = std::max(discr, 0.0);
        double eta = (a - (rho / std::abs(rho)) * std::sqrt(discr)) / (2.0 * c);
        lam_0 += eta;
        if (std::abs(eta) < tol * std::max(1e-8, std::abs(lam_0))) {
            break;
        }
    }
    return std::make_pair(shift + lam_0, lam_0);
}


double bisection(
    const std::function<double(double)>& f,
    double a,
    double b,
    const double tol,
    const unsigned int max_iter) {
    unsigned int iter_count = 0;

    while ((b - a) / 2.0 > tol) {
        double c = 0.5 * (a + b);
        if (std::abs(f(c)) < tol) {
            return c;
        } else if (f(a) * f(c) < 0.0) {
            b = c;
        } else {
            a = c;
        }
        ++iter_count;
        if (iter_count >= max_iter) {
            break;
        }
    }

    return 0.5 * (a + b);
}


double compute_outer_zero(
    const std::vector<double>& v,
    const std::vector<double>& d,
    const double rho,
    const double interval_end,
    const double tol = 1e-14,
    const unsigned int max_iter = 1000){

    const double threshold = 1e-11;
    double update = 0.0;

    // Compute L2 norm of v
    for (size_t i = 0; i < v.size(); ++i) {
        update += v[i] * v[i];
    }
    update = std::sqrt(update);

    auto f = [&](double x) -> double {
        double sum = 0.0;
        for (size_t k = 0; k < d.size(); ++k) {
            double denom = x - d[k];
            sum += (v[k] * v[k]) / denom;
        }
        return 1.0 - rho * sum;
    };

    double a, b;

    if (rho > 0.0) {
        a = interval_end + threshold;
        b = interval_end + 1.0;
        while (f(a) * f(b) > 0.0) {
            a = b;
            b += update;
        }
    } else if (rho < 0.0) {
        b = interval_end - threshold;
        a = interval_end - 1.0;
        while (f(a) * f(b) > 0.0) {
            b = a;
            a -= update;
        }
    }

    return bisection(f, a, b, tol, max_iter);
}


std::tuple<py::array_t<double>, py::array_t<int>, py::array_t<double>>
secular_solver(
    const double rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> d_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> v_np,
    py::array_t<unsigned int, py::array::c_style | py::array::forcecast> indices_np) {

    std::vector<double> d(d_np.data(), d_np.data() + d_np.size());
    std::vector<double> v(v_np.data(), v_np.data() + v_np.size());
    std::vector<unsigned int> indices(indices_np.data(), indices_np.data() + indices_np.size());

    std::vector<double> eig_val;
    std::vector<int> index;
    std::vector<double> delta;

    const unsigned int d_size = d.size();

    auto f = [&](double x) -> double {
        double sum = 0.0;
        for (size_t j = 0; j < v.size(); ++j) {
            sum += v[j] * v[j] / (x - d[j]);
        }
        return 1.0 - rho * sum;
    };

    if (rho > 0.0) {
        for (unsigned int i : indices) {
            if (i == d_size-1) { continue; }
            double lam_0 = 0.5 * (d[i] + d[i + 1]);
            bool left_center;
            if (f(lam_0) > 0.0) {
                left_center = true;
                index.push_back(static_cast<int>(i));
            } else {
                left_center = false;
                index.push_back(static_cast<int>(i + 1));
            }
            auto result = find_root(static_cast<int>(i), left_center, v, d, rho, lam_0);
            eig_val.push_back(result.first);
            delta.push_back(result.second);
        }
        if (indices.back() == d_size - 1)
        {
            double lam_0 = d[d.size() - 1];
            double Eig = compute_outer_zero(v, d, rho, lam_0);
            eig_val.push_back(Eig);
            index.push_back(static_cast<int>(d.size() - 1));
            delta.push_back(Eig - d[d.size() - 1]);
        }
    } else { // rho < 0.0
        if (indices[0] == 0)
        {
            double lam_0 = d[0];
            index.push_back(0);
            double Eig = compute_outer_zero(v, d, rho, lam_0);
            eig_val.push_back(Eig);
            delta.push_back(Eig - d[0]);
        }

        for (unsigned int i : indices) {
            if (i==0) continue;
            double lam_0 = 0.5 * (d[i-1] + d[i]);
            bool left_center;
            if (f(lam_0) > 0.0) {
                left_center = false;
                index.push_back(static_cast<int>(i + 1));
            } else {
                left_center = true;
                index.push_back(static_cast<int>(i));
            }
            eig_val.push_back(find_root(static_cast<int>(i)-1, left_center, v, d, rho, lam_0).first);
        }
    }
    
    py::array_t<double> eig_val_np(eig_val.size(), eig_val.data());
    py::array_t<int> index_np(index.size(), index.data());
    py::array_t<double> delta_np(delta.size(), delta.data());

    return std::make_tuple(eig_val_np, index_np, delta_np);
}


#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_set>

/**
 * Applies the deflation step in a divide-and-conquer eigenvalue algorithm.
 *
 * @param D          Diagonal entries of the matrix (as Eigen::VectorXd).
 * @param v          Rank-one update vector (modified in-place).
 * @param beta       Scalar multiplier for the rank-one update.
 * @param tol_factor Factor to scale the deflation tolerance (default 1e-12).
 * @return A tuple containing:
 *     - deflated_eigvals: Vector of trivial eigenvalues (Eigen::VectorXd).
 *     - deflated_eigvecs: Matrix whose columns are the trivial eigenvectors (Eigen::MatrixXd).
 *     - D_keep: Remaining diagonal entries after deflation (Eigen::VectorXd).
 *     - v_keep: Remaining rank-one vector entries after deflation (Eigen::VectorXd).
 *     - P_final:   Combined permutation & Givens rotation as an Eigen::SparseMatrix<double>.
 */


/**
 * Applies the deflation step in a divide-and-conquer eigenvalue algorithm.
 *
 * @param D          Diagonal entries of the matrix (as Eigen::VectorXd).
 * @param v          Rank-one update vector (modified in-place).
 * @param beta       Scalar multiplier for the rank-one update.
 * @param tol_factor Factor to scale the deflation tolerance (default 1e-12).
 * @return A tuple containing:
 *     - deflated_eigvals: Vector of trivial eigenvalues (Eigen::VectorXd).
 *     - deflated_eigvecs: Matrix whose columns are the trivial eigenvectors (Eigen::MatrixXd).
 *     - D_keep: Remaining diagonal entries after deflation (Eigen::VectorXd).
 *     - v_keep: Remaining rank-one vector entries after deflation (Eigen::VectorXd).
 *     - P_final: Combined permutation & rotation as an Eigen::SparseMatrix<double>.
 */
std::tuple<
    Eigen::VectorXd,
    Eigen::MatrixXd,
    Eigen::VectorXd,
    Eigen::VectorXd,
    Eigen::SparseMatrix<double>>
deflateEigenpairs(
    const Eigen::VectorXd& D,
    Eigen::VectorXd v,
    double beta,
    double tol_factor = 1e-12
) {
    int n = D.size();
    // 1) Build full matrix M and compute norm for tolerance
    Eigen::MatrixXd M = D.asDiagonal();          
    M += beta * v * v.transpose();               
    double norm_T = M.norm();                    
    double tol = tol_factor * norm_T;            

    // 2) Prepare containers for deflation
    std::vector<int> keep_indices, deflated_indices;
    Eigen::VectorXd deflated_eigvals = Eigen::VectorXd::Zero(n);
    std::vector<Eigen::VectorXd> deflated_eigvecs_list;
    int j = 0;  

    // 3) Zero-component deflation
    for (int i = 0; i < n; ++i) {
        if (std::abs(v(i)) < tol) {
            deflated_eigvals(j) = D(i);
            Eigen::VectorXd e_vec = Eigen::VectorXd::Zero(n);
            e_vec(i) = 1.0;
            deflated_eigvecs_list.push_back(e_vec);
            deflated_indices.push_back(i);
            ++j;
        } else {
            keep_indices.push_back(i);
        }
    }

    // 4) Build permutation P: [keep_indices, deflated_indices]
    std::vector<int> new_order;
    new_order.reserve(n);
    new_order.insert(new_order.end(), keep_indices.begin(), keep_indices.end());
    new_order.insert(new_order.end(), deflated_indices.begin(), deflated_indices.end());
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(n);
    P.indices() = Eigen::VectorXi::Map(new_order.data(), n);

    // 5) Extract subproblem D_keep and v_keep
    Eigen::VectorXd D_keep(static_cast<int>(keep_indices.size()));
    Eigen::VectorXd v_keep(static_cast<int>(keep_indices.size()));
    for (int idx = 0; idx < static_cast<int>(keep_indices.size()); ++idx) {
        D_keep(idx) = D(keep_indices[idx]);
        v_keep(idx) = v(keep_indices[idx]);
    }

    // 6) Givens rotations for near-duplicate entries
    std::unordered_set<int> to_check;
    to_check.reserve(keep_indices.size());
    for (int i = 0; i < static_cast<int>(keep_indices.size()); ++i)
        to_check.insert(i);
    std::vector<std::tuple<int,int,double,double>> rotations;
    std::vector<int> vec_idx_list;
    std::vector<int> to_check_copy(to_check.begin(), to_check.end());

    for (size_t idx_i = 0; idx_i + 1 < to_check_copy.size(); ++idx_i) {
        int i = to_check_copy[idx_i];
        if (to_check.find(i) == to_check.end()) continue;
        for (int k = i + 1; k < static_cast<int>(D_keep.size()); ++k) {
            if (std::abs(D_keep(k) - D_keep(i)) < tol) {
                to_check.erase(k);
                double r = std::hypot(v_keep(i), v_keep(k));
                double c = v_keep(i) / r;
                double s = -v_keep(k) / r;
                v_keep(i) = r;
                v_keep(k) = 0.0;
                rotations.emplace_back(i, k, c, s);
                deflated_eigvals(j) = D_keep(i);
                ++j;
                // local eigenvector in full basis
                Eigen::VectorXd tmp = Eigen::VectorXd::Zero(n);
                tmp(k) = c;
                tmp(i) = s;
                deflated_eigvecs_list.push_back(P.transpose() * tmp);
                vec_idx_list.push_back(k);
            }
        }
    }

    // 7) Final ordering after rotations
    std::vector<int> final_order(to_check.begin(), to_check.end());
    final_order.insert(final_order.end(), vec_idx_list.begin(), vec_idx_list.end());

    // 8) Resize deflated_eigvals to actual number found
    deflated_eigvals.conservativeResize(j);

    // 9) Build P2: accumulate sparse Givens rotations
    Eigen::SparseMatrix<double> P2(n,n);
    P2.setIdentity();
    for (auto &rot: rotations) {
        int i,k; double c,s; std::tie(i,k,c,s) = rot;
        Eigen::SparseMatrix<double> G(n,n);
        G.setIdentity();
        G.coeffRef(i,i) = c;
        G.coeffRef(k,k) = c;
        G.coeffRef(i,k) = -s;
        G.coeffRef(k,i) = s;
        P2 = P2 * G;
    }

    // 10) Build permutation matrix P3 from final_order
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P3(n);
    P3.indices() = Eigen::VectorXi::Map(final_order.data(), n);

    // 11) Convert P and P3 to sparse<double> and combine all transforms
    Eigen::SparseMatrix<double> P_sparse  = P.toDenseMatrix().cast<double>().sparseView();
    Eigen::SparseMatrix<double> P3_sparse = P3.toDenseMatrix().cast<double>().sparseView();
    Eigen::SparseMatrix<double> P_final   = P3_sparse * P2 * P_sparse;

    // 12) Extract final D_keep and v_keep for reduced problem
    std::vector<int> final_keep(to_check.begin(), to_check.end());
    Eigen::VectorXd D_keep_final(static_cast<int>(final_keep.size()));
    Eigen::VectorXd v_keep_final(static_cast<int>(final_keep.size()));
    for (int idx = 0; idx < static_cast<int>(final_keep.size()); ++idx) {
        D_keep_final(idx) = D_keep(final_keep[idx]);
        v_keep_final(idx) = v_keep(final_keep[idx]);
    }

    // 13) Assemble deflated eigenvectors matrix
    Eigen::MatrixXd deflated_eigvecs(n, static_cast<int>(deflated_eigvecs_list.size()));
    for (int col = 0; col < static_cast<int>(deflated_eigvecs_list.size()); ++col) {
        deflated_eigvecs.col(col) = deflated_eigvecs_list[col];
    }

    return std::make_tuple(
        deflated_eigvals,
        deflated_eigvecs.transpose(),
        D_keep_final,
        v_keep_final,
        P_final
    );
}


// PYTHON BINDINGS USING PYBIND11

PYBIND11_MODULE(cxx_utils, m) {
    m.doc() = "Function that computes the eigenvalue and eigenvector"; // Optional module docstring.

    m.def("QR_algorithm", &QR_algorithm, py::arg("diag"), py::arg("off_diag"), py::arg("tol")=1e-8, py::arg("max_iter")=5000);
    m.def("Eigen_value_calculator", &Eigen_value_calculator, py::arg("diag"), py::arg("off_diag"), py::arg("tol")=1e-8, py::arg("max_iter")=5000);
    m.def("secular_solver_cxx", &secular_solver, py::arg("rho"), py::arg("d"), py::arg("v"), py::arg("indices"));
    m.def("deflate_eigenpairs_cxx", &deflateEigenpairs, py::arg("D"), py::arg("v"), py::arg("beta"), py::arg("tol_factor") = 1e-12);
}