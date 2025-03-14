#include <iostream>
#include <utility>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
	


std::pair<std::vector<double>, std::vector<std::vector<double>> > 
 QR_algorithm(std::vector<double>  diag, std::vector<double>  off_diag, const double toll=1e-8, const unsigned int max_iter=5000){

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
    double toll_equivalence=1e-10;
    double w=0, z=0;

    

    while (iter<max_iter && m>0)
    {
        // prefetching most used value to avoid call overhead
        a_m=diag[m];
        b_m_1=off_diag[m-1];
        d=(diag[m-1]-a_m)*0.5;
        
        if(std::abs(d)<toll_equivalence){
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
                if(std::abs(d)<toll_equivalence){
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
        for(unsigned int i=0; i<n-1; i++){
            c=Matrix_trigonometric[i][0];
            s=Matrix_trigonometric[i][1];
            
            for(j=0; j<n;j=j+5){
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
        if ( std::abs(off_diag[m-1]) < toll*( std::abs(diag[m]) + std::abs(diag[m-1]) )  )
        {
            --m;
        }
    }

    if(iter==max_iter){
        std::cout<<"Converges failed"<<std::endl;
    }

    std::vector<std::vector<double>> eig_vec(n,std::vector<double> (n, 0));
    //std::cout<<"Iteration: "<<iter<<std::endl;
    for(unsigned int i=0; i<n; i++){
        for(unsigned j=0; j<n; j++){
            eig_vec[i][j]=Q[i+j*n];
        }
    }

    return std::make_pair(diag, eig_vec);
 

}

std::vector<double>
 Eigen_value_calculator(std::vector<double>  diag, std::vector<double>  off_diag, const double toll=1e-8, const unsigned int max_iter=5000){

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
    double toll_equivalence=1e-10;
    double w=0, z=0;

    

    while (iter<max_iter && m>0)
    {
        // prefetching most used value to avoid call overhead
        a_m=diag[m];
        b_m_1=off_diag[m-1];
        d=(diag[m-1]-a_m)*0.5;
        
        if(std::abs(d)<toll_equivalence){
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
                if(std::abs(d)<toll_equivalence){
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
        if ( std::abs(off_diag[m-1]) < toll*( std::abs(diag[m]) + std::abs(diag[m-1]) )  )
        {
            --m;
        }
    }

    if(iter==max_iter){
        std::cout<<"Converges failed"<<std::endl;
    }


    return diag;
 

}

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"



namespace py=pybind11;

PYBIND11_MODULE(QR_cpp, m) {
    m.doc() = "Function that computes the eigenvalue and eigenvector"; // Optional module docstring.

    m.def("QR_algorithm", &QR_algorithm, py::arg("diag"), py::arg("off_diag"), py::arg("toll")=1e-8, py::arg("max_iter")=5000);
    m.def("Eigen_value_calculator", &Eigen_value_calculator, py::arg("diag"), py::arg("off_diag"), py::arg("toll")=1e-8, py::arg("max_iter")=5000);
}