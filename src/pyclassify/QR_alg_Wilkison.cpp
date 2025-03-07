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

std::vector<std::vector<double>> 
GivensProduct(const std::vector<std::array<double, 2>>& Matrix_trigonometric,
              unsigned int start,
              unsigned int end,
              unsigned int n)
{
    // Create identity matrix Q of size n x n.
    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0.0));
    for (unsigned int row = 0; row < n; ++row) {
        Q[row][row] = 1.0;
    }

    // For each rotation from start to end-1, update columns i and i+1.
    for (unsigned int i = start; i < end; ++i)
    {
        double c = Matrix_trigonometric[i][0];
        double s = Matrix_trigonometric[i][1];

        // Rotate columns i and i+1 in Q (parallelizing over rows)
        #pragma omp parallel for
        for (int row = start; row < i+2; ++row)
        {
            double tmp = Q[row][i];
            Q[row][i]   = c * tmp - s * Q[row][i + 1];
            Q[row][i+1] = s * tmp + c * Q[row][i + 1];
        }
    }
    return Q;
}

void operator/=(std::vector<double> & x, const double scale_factor){
    std::for_each(x.begin(), x.end(), [scale_factor](double element){return element/scale_factor;});
}
std::vector<double> operator/(const std::vector<double> & x, const double scale_factor){
    std::vector<double> z;
    std::transform(x.cbegin(), x.cend(), std::front_inserter(z), [scale_factor](double element){return element/scale_factor;});
    return z;
}
std::vector<double> operator*(std::vector<double> & x, const double scale_factor){
    std::for_each(x.begin(), x.end(), [scale_factor](double element){return element*scale_factor;});
    return x;
}

double norm_2(std::vector<double> & x){
    double sum=std::accumulate(x.begin(), x.end(), 0, [](double a, double b) { return a + b*b; });
    return std::sqrt(sum);
}
void operator-=(std::vector<double> & x, const std::vector<double>& y){
    std::transform(x.begin(), x.end(), y.begin(), x.begin(), std::minus<double>());
}
std::vector<double> operator-(std::vector<double> & x, const std::vector<double>& y){
    std::vector<double> z;
    std::transform(x.begin(), x.end(), y.begin(), std::front_inserter(z), std::minus<double>());
    return z;
}
std::vector<double> operator*(const std::vector<std::vector<double>> & A, const std::vector<double> & b){
    std::vector<double> x(A.size(), 0);

    // Parallelize this  loop
    for(unsigned int i=0; i<A.size(); ++i){
        x[i]=std::inner_product(A[i].begin(), A[i].end(), b.begin(), 0);
    }
    return x;
}

double operator*(const std::vector<double> & x, const std::vector<double> & y){

    return std::inner_product(x.cbegin(), x.cend(), y.cbegin(), 0);

}

void Lanczos_PRO(std::vector<std::vector<double>> A, std::vector<double> q, const unsigned int m, const double toll=1e-6){

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

        }

            return Q, alpha, beta[:-1]    


    }
    


}


//std::pair<std::vector<double>, std::vector<std::vector<double>> > 
void QR_algorithm(std::vector<double>  diag, std::vector<double>  off_diag, const double toll=1e-8, const unsigned int max_iter=1000){

    const unsigned int n = diag.size();

    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0));

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
    int m=n-1;
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




        //Uncomment to compute the eigenvalue
        for(unsigned int i=0; i<n-1; i++){
            for(unsigned j=0; j<n;j++){
                tmp=Q[j][i];
                Q[j][i]=tmp*Matrix_trigonometric[i][0]-Q[j][i+1]*Matrix_trigonometric[i][1];
                Q[j][i+1]=tmp*Matrix_trigonometric[i][1]+Q[j][i+1]*Matrix_trigonometric[i][0];
            }
        }
    

        unsigned int n_processor, delta_i=n/n_processor;
        std::future<std::vector<std::vector<double>>> future1 = std::async(std::launch::async, GivensProduct, Matrix_trigonometric, 0, 250 - 1, 500);
        std::future<std::vector<std::vector<double>>> future2 = std::async(std::launch::async, GivensProduct, Matrix_trigonometric, 250, 500 - 1 ,500);

        std::vector<  std::future<std::vector<std::vector<double>>> > vector_thread;
        std::vector <unsigned int> index;
        for(unsigned int i=0;i<n_processor;i++){
            index.push_back(i*delta_i);
        }

        index.push_back(n-1);

        iter++;
        if ( std::abs(off_diag[m-1]) < toll*( std::abs(diag[m]) + std::abs(diag[m-1]) )  )
        {
            --m;
        }
    }



    

}


int main(){

    std::vector<double> diag(500, 5), offdiag(499, 20);

    QR_algorithm(diag, offdiag);
    
    auto start = std::chrono::high_resolution_clock::now();

    QR_algorithm(diag, offdiag);

    // Capture the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Compute the elapsed time as a duration
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Elapsed time: " << elapsed.count() << " milliseconds" << std::endl;
    return 0;
}