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

std::vector<std::vector<double>> 
GivensProduct(const std::vector<std::array<double, 2>> Matrix_trigonometric,
              unsigned int start,
              unsigned int end,
              unsigned int n)
{
    // Determine the size of the nontrivial block.
    unsigned int m = end - start + 1;
    

    // Allocate block B as an m x m matrix stored by column.
    // B[j][r] is element at row r in column j.
    std::vector<std::vector<double>> B(m, std::vector<double>(m, 0.0));

    // Initialize B as the identity (in column–major form).
    for (unsigned int j = 0; j < m; ++j) {
        B[j][j] = 1.0;
    }

    // For each rotation in the range [start, end), reindexed by k = i - start.
    // Each rotation affects columns k and k+1, and only the first (k+2) rows.
    for (unsigned int k = 0; k < m - 1; ++k) {
        double c = Matrix_trigonometric[k][0];
        double s = Matrix_trigonometric[k][1];

        for (unsigned int r = 0; r < k + 2; ++r) {
            double tmp = B[k][r];
            B[k][r]   = c * tmp - s * B[k+1][r];
            B[k+1][r] = s * tmp + c * B[k+1][r];
        }
    }

    return B;
}

// Function that processes a block of columns [i_start, i_end)
void Mat_Mat_mul(const std::vector<std::array<double, 2>>& local_trig,
    const std::vector<std::vector<double>>& Q,
    std::vector<std::vector<double>>& Q_posterior,
    unsigned int i_start, unsigned int i_end,
    unsigned int n) {
    // Note: local_trig.size() should equal (i_end - i_start)
    for (unsigned int i = i_start, local_i = 0; i < i_end; ++i, ++local_i) {
        // Extract c and s from the local slice
        double c = local_trig[local_i][0];
        double s = local_trig[local_i][1];

        // Process the inner loop in blocks of 4
        for (unsigned int j = 0; j < n; j += 2) {
            // Block for Q[j][i] and Q[j][i+1]
            double tmp = Q[j][i];
            Q_posterior[j][i]   = tmp * c - Q[j][i+1] * s;
            Q_posterior[j][i+1] = tmp * s + Q[j][i+1] * c;

            // Next block for Q[j+1][i] and Q[j+1][i+1]
            tmp = Q[j+1][i];
            Q_posterior[j+1][i]   = tmp * c - Q[j+1][i+1] * s;
            Q_posterior[j+1][i+1] = tmp * s + Q[j+1][i+1] * c;

    }
    }
}

// void operator/=(std::vector<double> & x, const double scale_factor){
//     std::for_each(x.begin(), x.end(), [scale_factor](double element){return element/scale_factor;});
// }
// std::vector<double> operator/(const std::vector<double> & x, const double scale_factor){
//     std::vector<double> z;
//     std::transform(x.cbegin(), x.cend(), std::front_inserter(z), [scale_factor](double element){return element/scale_factor;});
//     return z;
// }
std::vector<double> operator*(std::vector<double> & x, const double scale_factor){
    std::for_each(x.begin(), x.end(), [scale_factor](double element){return element*scale_factor;});
    return x;
}

// double norm_2(std::vector<double> & x){
//     double sum=std::accumulate(x.begin(), x.end(), 0, [](double a, double b) { return a + b*b; });
//     return std::sqrt(sum);
// }
// void operator-=(std::vector<double> & x, const std::vector<double>& y){
//     std::transform(x.begin(), x.end(), y.begin(), x.begin(), std::minus<double>());
// }
// std::vector<double> operator-(std::vector<double> & x, const std::vector<double>& y){
//     std::vector<double> z;
//     std::transform(x.begin(), x.end(), y.begin(), std::front_inserter(z), std::minus<double>());
//     return z;
// }
// std::vector<double> operator*(const std::vector<std::vector<double>> & A, const std::vector<double> & b){
//     std::vector<double> x(A.size(), 0);

//     // Parallelize this  loop
//     for(unsigned int i=0; i<A.size(); ++i){
//         x[i]=std::inner_product(A[i].begin(), A[i].end(), b.begin(), 0);
//     }
//     return x;
// }

// double operator*(const std::vector<double> & x, const std::vector<double> & y){

//     return std::inner_product(x.cbegin(), x.cend(), y.cbegin(), 0);

// }

// void Lanczos_PRO(std::vector<std::vector<double>> A, std::vector<double> q, const unsigned int m, const double toll=1e-6){

//     q/=norm_2(q);    
//     std::vector<std::vector<double>> Q{q};

//     std::vector<double> r=A*q;

//     std::vector<double> alpha, beta;

//     alpha.push_back(q*r);
//     r=r-q*alpha[0];

//     beta.push_back(norm_2(r));
//     std::vector<double> res;

//     for (unsigned int j = 1; j < m; j++)
//     {
//         q= r/beta[j-1];
//         res= Q*q;

//         for(auto const ele: res){
//             if(std::abs(ele)>toll){
                
//                 for(unsigned int i=0;i<Q.size(); ++i){
//                     double h=(q*Q[i]);
//                     q=q-Q[i]*h;
//                 }
//                 break;
//             }
//         }
//         q/=norm_2(q);
//         Q.push_back(q);
//         r=A*q-(Q[j-1]*beta[j-1]); 
//         alpha.push_back(q * r);
//         r = r -  q *alpha[j];
//         beta.push_back(norm_2(r));

//         if (std::abs(beta[j]) < 1e-15){
//             //return Q, alpha, beta[:-1]
//             break;
//         }

                


//     }
    


// }

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


//std::pair<std::vector<double>, std::vector<std::vector<double>> > 
void QR_algorithm(std::vector<double>  diag, std::vector<double>  off_diag, const double toll=1e-8, const unsigned int max_iter=10000){

    const unsigned int n = diag.size();

    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0)), R(n);

    for(unsigned int i = 0; i < n; i++){
        Q[i][i] = 1;
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


    // unsigned int n_processor=8, delta_i=n/n_processor;
    // // // std::future<std::vector<std::vector<double>>> future1 = std::async(std::launch::async, GivensProduct, Matrix_trigonometric, 0, 250, 500);
    // // // std::future<std::vector<std::vector<double>>> future2 = std::async(std::launch::async, GivensProduct, Matrix_trigonometric, 250, 500 - 1 ,500);

    // std::vector<  std::future<std::vector<std::vector<double>>> > vector_thread(n_processor);
    // std::vector <unsigned int> index;

    // std::vector<  std::future< void > > vec_matrix_mult(n_processor);
    
    // for(unsigned int i=0;i<n_processor;i++){
    //     index.push_back(i*delta_i);
    // }

    // index.push_back(n-1);

    // std::vector<double> line0(n), line1(n), line3(n), line2(n);

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



        unsigned j;
        //Uncomment to compute the eigenvalue
        //collapse(2) 
        for(unsigned int i=0; i<n-1; i++){
            //std::vector<std::vector<double>> Q_posterior(Q);
            c=Matrix_trigonometric[i][0];
            s=Matrix_trigonometric[i][1];
            #pragma omp parallel for
            for(j=0; j<n;j=j+5){
                tmp=Q[i][j];
                Q[i][j]=tmp*c-Q[i+1][j]*s;
                Q[i+1][j]=tmp*s+Q[i+1][j]*c;
                tmp=Q[i][j+1];
                Q[i][j+1]=tmp*c-Q[i+1][j+1]*s;
                Q[i+1][j+1]=tmp*s+Q[i+1][j+1]*c;
                tmp=Q[i][j+2];
                Q[i][j+2]=tmp*c-Q[i+1][j+2]*s;
                Q[i+1][j+2]=tmp*s+Q[i+1][j+2]*c;
                tmp=Q[i][j+3];
                Q[i][j+3]=tmp*c-Q[i+1][j+3]*s;
                Q[i+1][j+3]=tmp*s+Q[i+1][j+3]*c;               
                tmp=Q[i][j+4];
                Q[i][j+4]=tmp*c-Q[i+1][j+4]*s;
                Q[i+1][j+4]=tmp*s+Q[i+1][j+4]*c;
            }
            for(; j < n; j++)
            {
                tmp=Q[i][j];
                Q[i][j]=tmp*c-Q[i+1][j]*s;
                Q[i+1][j]=tmp*s+Q[i+1][j]*c;
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
    std::cout<<"Iteration: "<<iter<<std::endl;


    

}


int main(){
    std::vector<double> diag(2000, 5), offdiag(1999, 20);


    
    auto start = std::chrono::high_resolution_clock::now();

    QR_algorithm(diag, offdiag, 1e-8, 50000);

    // Capture the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Compute the elapsed time as a duration
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Elapsed time: " << elapsed.count() << " milliseconds" << std::endl;
    return 0;
}
