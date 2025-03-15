#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>
#include <iomanip>
#include <chrono>
#include <future>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <random>
#include <execution>


void operator/=(std::vector<double> & x, const double scale_factor){
    std::for_each(std::execution::par, x.begin(), x.end(), [scale_factor](double & element){return element /= scale_factor;;});
}
std::vector<double> operator/(const std::vector<double> & x, const double scale_factor){
    std::vector<double> z;
    std::transform(x.cbegin(), x.cend(), std::back_inserter(z), [scale_factor](double element){return element/scale_factor;});
    return z;
}
std::vector<double> operator*(std::vector<double>  x, const double scale_factor){
    std::for_each(std::execution::par,  x.begin(), x.end(), [scale_factor](double & element){return element *= scale_factor;});
    return x;
}

double norm_2(const std::vector<double> & x){
    double sum=std::accumulate(x.begin(), x.end(), 0.0, [](double a, double b) { return a + b*b; });
    return std::sqrt(sum);
}
void operator-=(std::vector<double> & x, const std::vector<double>& y){
    std::transform(std::execution::par, x.begin(), x.end(), y.begin(), x.begin(), std::minus<double>());
}
std::vector<double> operator-(const std::vector<double> & x, const std::vector<double>& y){
    std::vector<double> z(x.size());
    std::transform(std::execution::par, x.begin(), x.end(), y.begin(), z.begin(), std::minus<double>());
    return z;
}
std::vector<double> operator*(const std::vector<std::vector<double>> & A, const std::vector<double> & b){
    std::vector<double> x(A.size(), 0);

    // Parallelize this  loop
    //#pragma omp parallel for //collapse(2)
    for(unsigned int i=0; i<A.size(); ++i){
        x[i]=std::inner_product(A[i].begin(), A[i].end(), b.begin(), 0.0);
        // for(unsigned int j=0; j<x.size(); j++){
        //     x[i]  += A[i][j]*b[j];
        // }
    }
    return x;
}

double operator*(const std::vector<double> & x, const std::vector<double> & y){

    // double sum=0;
    // #pragma omp parallel for
    // for(unsigned int j=0; j<x.size(); j++){
    //     sum += x[j]*y[j];
    // }
    return std::inner_product(x.cbegin(), x.cend(), y.cbegin(), 0.0);

}

std::tuple<std::vector<std::vector<double>>, std::vector<double>, std::vector<double> > 
Lanczos_PRO(std::vector<std::vector<double>> A, std::vector<double> q, const unsigned int m, const double toll=1e-6){

    q/=norm_2(q);    
    std::vector<std::vector<double>> Q{q};

    std::vector<double> r=A*q;

    std::vector<double> alpha, beta;

    alpha.push_back(q*r);
    r=r-q*alpha[0];

    beta.push_back(norm_2(r));
    std::vector<double> res;

    for (unsigned int j = 1; j < m; j++)
    {
        q= r/beta[j-1];
        res= Q*q;
       
        for(auto const ele: res){
            if(std::abs(ele)>toll){
                //#pragma omp parallel for
                for(unsigned int i=0;i<Q.size(); ++i){
                    double h=(q*Q[i]);
                    q=q-Q[i]*h;
                }
                break;
            }
        }
        q/=norm_2(q);
        Q.push_back(q);
        r=A*q-(Q[j-1]*beta[j-1]); 
        alpha.push_back(q * r);
        r = r -  q *alpha[j];
        beta.push_back(norm_2(r));

        if (std::abs(beta[j]) < 1e-15){
            beta.pop_back();
            return std::make_tuple( Q, alpha, beta);
            break;
        }

                


    }
    
    beta.pop_back();
    return std::make_tuple( Q, alpha, beta);


}

void print_matrix(const std::vector<std::vector<double> > & Q){
    for(unsigned int i=0; i<Q.size(); i++){
        for(unsigned j=0; j<Q.size(); j++){
            std::cout<<std::setw(10)<<Q[j][i];
        }
        std::cout<<"\n";
    }
    std::cout<<"--------------------------------------------------------"<<"\n";

    std::cout<<"------------------------------------------------" <<"\n";

}


std::pair<std::vector<double>, std::vector<std::vector<double>> > 
 QR_algorithm(std::vector<double>  diag, std::vector<double>  off_diag, const double toll=1e-8, const unsigned int max_iter=10000){

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

int main(){

    // std::vector<double> diag(400, 5), offdiag(399, 20);

    
    // auto start = std::chrono::high_resolution_clock::now();

    // QR_algorithm(diag, offdiag, 1e-8, 50000);

    // // Capture the end time
    // auto end = std::chrono::high_resolution_clock::now();

    // // Compute the elapsed time as a duration
    // auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // std::cout << "Elapsed time: " << elapsed.count() << " milliseconds" << std::endl;

    std::vector<std::vector<double>> matrix_test = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        {2, 11, 12, 13, 14, 15, 16, 17, 18, 19},
        {3, 12, 20, 21, 22, 23, 24, 25, 26, 27},
        {4, 13, 21, 28, 29, 30, 31, 32, 33, 34},
        {5, 14, 22, 29, 35, 36, 37, 38, 39, 40},
        {6, 15, 23, 30, 36, 41, 42, 43, 44, 45},
        {7, 16, 24, 31, 37, 42, 46, 47, 48, 49},
        {8, 17, 25, 32, 38, 43, 47, 50, 51, 52},
        {9, 18, 26, 33, 39, 44, 48, 51, 53, 54},
        {10, 19, 27, 34, 40, 45, 49, 52, 54, 55}
    };
    Lanczos_PRO(matrix_test, std::vector<double> (10, 1), 10);
    const int n = 3000;
    // Create a 2D vector (matrix) initialized to 0.0
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));

    // Initialize random number generator with a random seed
    std::mt19937 rng(std::random_device{}());
    // Define a uniform distribution for random values (adjust range as needed)
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Fill the matrix ensuring symmetry
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            double value = dist(rng);
            matrix[i][j] = value;
            matrix[j][i] = value;  // mirror the value to maintain symmetry
        }
    }


    std::vector<double> initial_guess(n, 1);

    auto start = std::chrono::high_resolution_clock::now();
    Lanczos_PRO(matrix, initial_guess, n);
    auto end = std::chrono::high_resolution_clock::now();

    // Compute the elapsed time as a duration
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << elapsed.count() << " milliseconds" << std::endl;

    return 0;
}