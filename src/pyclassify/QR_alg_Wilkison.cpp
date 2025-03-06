#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>


//std::pair<std::vector<double>, std::vector<std::vector<double>> > 
void QR_algorithm(std::vector<double>  diag, std::vector<double>  off_diag, const double toll=1e-8, const unsigned int max_iter=1000){

    const unsigned int n = diag.size();

    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0));

    for(unsigned int i = 1; i < n-1; i++){
        Q[i][i] = 1;
    }
    Q[0][0] = 1;
    Q[n-1][n-1] = 1;


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
                        x_new=x_0+std::cos(x_0)*std::cos(x_0)*(std::tan(x_0) + b_2/d);
                        err_rel=std::abs((x_new-x_0)/x_new);
                        x_0=x_new;
                        ++iter_newton;
                    }
                    c=std::cos(x_new/2);
                    s=std::sin(x_new/2);
                    x=x+mu;

                }

            }

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
            
        }
        iter++;
        if ( std::abs(off_diag[m-1]) < toll*( std::abs(diag[m]) + std::abs(diag[m-1]) )  )
        {
            --m;
        }
        
        
    }
    

}


int main(){

    std::vector<double> diag{1, 2, 3, 4, 5}, offdiag(4, 2);
    QR_algorithm(diag, offdiag);
    return 0;
}