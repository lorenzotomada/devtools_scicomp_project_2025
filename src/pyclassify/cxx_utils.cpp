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


std::pair<double, double> find_root(
    const unsigned int i,
    const bool left_center,
    const std::vector<double>& v,
    const std::vector<double>& d,
    const double rho,
    double lam_0,
    const double tol = 1e-12,
    const unsigned int maxiter = 100) {
    std::vector<double> diag = d;
    double shift = 0.0;
    if (left_center) {
        const double di = d[i];
        for (double& x : diag) x -= di;
        lam_0 -= di;
        shift = di;
    } else {
        const double di1 = d[i + 1];
        for (double& x : diag) x -= di1;
        lam_0 -= di1;
        shift = di1;
    }

    auto Psi_1 = [&](const double x) -> double {
        double sum = 0.0;
        for (unsigned int j = 0; j <= i; ++j)
            sum += v[j] * v[j] / (diag[j] - x);
        return rho * sum;
    };
    auto Psi_2 = [&](const double x) -> double {
        double sum = 0.0;
        for (unsigned int j = i + 1; j < diag.size(); ++j)
            sum += v[j] * v[j] / (diag[j] - x);
        return rho * sum;
    };
    auto dPsi_1 = [&](const double x) -> double {
        double sum = 0.0;
        for (unsigned int j = 0; j <= i; ++j)
            sum += v[j] * v[j] / ((diag[j] - x) * (diag[j] - x));
        return rho * sum;
    };
    auto dPsi_2 = [&](const double x) -> double {
        double sum = 0.0;
        for (unsigned int j = i + 1; j < diag.size(); ++j)
            sum += v[j] * v[j] / ((diag[j] - x) * (diag[j] - x));
        return rho * sum;
    };

    for (unsigned int iter = 0; iter < maxiter; ++iter) {
        const double delta_i = diag[i] - lam_0;
        const double delta_i1 = (i + 1 < diag.size()) ? diag[i + 1] - lam_0 : 0.0;
        const double vPsi_1 = Psi_1(lam_0);
        const double vPsi_2 = Psi_2(lam_0);
        const double vdPsi_1 = dPsi_1(lam_0);
        const double vdPsi_2 = dPsi_2(lam_0);

        const double a = (1.0 + vPsi_1 + vPsi_2) * (delta_i + delta_i1) - (vdPsi_1 + vdPsi_2) * delta_i * delta_i1;
        const double b = delta_i * delta_i1 * (1.0 + vPsi_1 + vPsi_2);
        const double c = 1.0 + vPsi_1 + vPsi_2 - delta_i * vdPsi_1 - delta_i1 * vdPsi_2;

        double discr = a * a - 4.0 * b * c;
        discr = std::max(discr, 0.0);

        const double eta = (a - rho / std::abs(rho) * std::sqrt(discr)) / (2.0 * c);
        lam_0 += eta;
        if (std::abs(eta) < tol * std::max(1e-6, std::abs(lam_0)))
            break;
    }

    return std::make_pair(shift + lam_0, lam_0);
}


double out_range(
    const std::vector<double>& v,
    const std::vector<double>& d,
    const double rho,
    double lam_0,
    const double tol = 1e-12,
    const unsigned int maxiter = 100) {
    std::vector<double> diag = d;
    double shift = 0.0;
    double d_i = 0.0;

    if (rho < 0) {
        for (auto& x : diag) x -= d[0];
        lam_0 -= d[0];
        shift = d[0];
        d_i = diag[0];
    } else {
        for (auto& x : diag) x -= d.back();
        lam_0 -= d.back();
        shift = d.back();
        d_i = diag.back();
    }

    auto Psi_2 = [&](const double x) -> double {
        double sum = 0.0;
        for (unsigned int j = 0; j < diag.size(); ++j)
            sum += v[j] * v[j] / (diag[j] - x);
        return rho * sum;
    };
    auto dPsi_2 = [&](const double x) -> double {
        double sum = 0.0;
        for (unsigned int j = 0; j < diag.size(); ++j)
            sum += v[j] * v[j] / ((diag[j] - x) * (diag[j] - x));
        return rho * sum;
    };

    for (unsigned int iter = 0; iter < maxiter; ++iter) {
        const double c_1 = dPsi_2(lam_0) * (d_i - lam_0) * (d_i - lam_0);
        const double c_3 = Psi_2(lam_0) - dPsi_2(lam_0) * (d_i - lam_0) + 1.0;
        const double lam = d_i + c_1 / c_3;
        if (std::abs(lam_0 - lam) < tol * std::max(1.0, std::abs(lam)))
            break;
        lam_0 = lam;
    }
    return shift + lam_0;
}



std::tuple<py::array_t<double>, std::vector<unsigned int>, py::array_t<double>>
secular_solver(const double rho,
               const std::vector<double>& d,
               const std::vector<double>& v//,
               //const std::vector<unsigned int>& indices
               ) {

    std::vector<double> eig_val;
    std::vector<unsigned int> index;
    std::vector<double> delta;

    auto f = [&](double x) -> double {
        double sum = 0.0;
        for (unsigned int j = 0; j < d.size(); ++j)
        {
            sum += (v[j] * v[j]) / (x - d[j]);
        }
        return 1.0 - rho * sum;
    };

    const unsigned int d_size = d.size();
    if (rho > 0.0)
    {
        for  (unsigned int i=0; i< d_size-1;++i) //(unsigned int i : indices)
        {
            if (i == d_size-1) { continue; }
            double lam_0 = 0.5 * (d[i] + d[i + 1]);
            bool left_center;

            if (f(lam_0) > 0.0)
            {
                left_center = true;
                index.push_back(i);
            }
            else
            {
                left_center = false;
                index.push_back(i + 1);
            }

            double Eig, Delta;
            std::tie(Eig, Delta) = find_root(i, left_center, v, d, rho, lam_0);
            eig_val.push_back(Eig);
            delta.push_back(Delta);
        }

        //if (indices.back() == d_size - 1)
        {
            double lam_0 = d[d.size() - 1] + 5.0 * (d[d.size() - 1] - d[d.size() - 2]);
            // bool left_center = false;
            index.push_back(static_cast<unsigned int>(d.size() - 1));
            double Eig = out_range(v, d, rho, lam_0);
            eig_val.push_back(Eig);
            delta.push_back(Eig - d[d.size() - 1]);
        }
    }
    else
    {
         
        //if (indices[0] == 0)
        {
            double lam_0 = d[0] - 5.0 * (d[1] - d[0]);
            // bool left_center = false;
            index.push_back(0);
            double Eig = out_range(v, d, rho, lam_0);
            eig_val.push_back(Eig);
            delta.push_back(Eig - d[0]);
        }

        for (unsigned int i =1; i<=d_size-1;++i) //(unsigned int i : indices)
        {   
            if (i == 0) { continue; }
            double lam_0 = 0.5 * (d[i] + d[i + 1]);
            bool left_center;

            if (f(lam_0) > 0.0)
            {
                left_center = false;
                index.push_back(i + 1);
            }
            else
            {
                left_center = true;
                index.push_back(i);
            }

            double Eig, dummy;
            std::tie(Eig, dummy) = find_root(i, left_center, v, d, rho, lam_0);
            eig_val.push_back(Eig);
        }
    }

    py::array_t<double> eig_val_np(eig_val.size(), eig_val.data());
    py::array_t<double> delta_np(delta.size(), delta.data());

    return std::make_tuple(eig_val_np, index, delta_np);
}


// PYTHON BINDINGS USING PYBIND11

PYBIND11_MODULE(cxx_utils, m) {
    m.doc() = "Function that computes the eigenvalue and eigenvector"; // Optional module docstring.

    m.def("QR_algorithm", &QR_algorithm, py::arg("diag"), py::arg("off_diag"), py::arg("tol")=1e-8, py::arg("max_iter")=5000);
    m.def("Eigen_value_calculator", &Eigen_value_calculator, py::arg("diag"), py::arg("off_diag"), py::arg("tol")=1e-8, py::arg("max_iter")=5000);
    m.def("secular_solver_cxx", &secular_solver, py::arg("rho"), py::arg("d"), py::arg("v"));//, py::arg("indices"));
}