use ndarray::{concatenate, Array1, Array2, Axis};
use ndarray_linalg::Norm;

fn compute(lambda_l: &Array1<f64>, lambda_w: &Array1<f64>, l: &Array1<f64>, w: &Array1<f64>, t: &Array2<f64>, k: i32) -> (Array1<f64>, Array1<f64>) {
    if k % 2 == 0 {
        (
            lambda_w.clone(),
            ((-lambda_w - 1. - t).mapv(f64::exp).t().sum_axis(Axis(0)) / l).mapv(f64::ln)
        )
    }
    else {
        (
            ((-lambda_l - 1. - t.t()).mapv(f64::exp).t().sum_axis(Axis(0)) / w).mapv(f64::ln),
            lambda_l.clone()
        )
    }
} 

fn rec_d_i_j(l: &Array1<f64>, w: &Array1<f64>, t: &Array2<f64>, peoples: i32, n :usize) -> Array2<f64> {
    (-1. - t - (Array2::from_shape_vec((n, 1), l.to_vec()).unwrap() + w)).mapv(f64::exp) * peoples as f64
}

pub fn sinkhorn(l: &Array1<f64>, w: &Array1<f64>, t: &Array2<f64>, max_iter: i32, eps: f64, peoples: i32) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    assert_eq!(l.len(), w.len());
    let n = l.len();
    let mut lambda_w = Array1::<f64>::zeros(n);
    let mut lambda_l = Array1::<f64>::zeros(n);

    for k in 0..max_iter {
        let (l_w, l_l) = compute(&lambda_l, &lambda_w, l, w, t, k);

        let delta = Norm::norm(&concatenate![Axis(0), &lambda_l - &l_l, &lambda_w - &l_w]);

        if delta < eps {
            break;
        }
        lambda_w = l_w;
        lambda_l = l_l;
        
    }
    let r = rec_d_i_j(&lambda_l, &lambda_w, t, peoples, n);
    (r, lambda_l, lambda_w)
}

#[cfg(test)]
mod test_sinkhorn {
    use ndarray::{array};

    use super::*;

    #[test]
    fn test() {
        let max_iter = 25000;
        let eps = 1e-8;
    
        let mut productions_vector = array![100., 200.];
        let mut attractions_vector = array![150., 150.];

        let peoples_num = productions_vector.sum();

        productions_vector /= peoples_num;
        attractions_vector /= peoples_num;

        let cost_matrix = array![
            [1., 2.],
            [2., 1.]];

        let (correspondence, _, _) = sinkhorn(&productions_vector, &attractions_vector, &cost_matrix, max_iter, eps, peoples_num as i32);

        print!("{:?}", correspondence);

        assert!(true);
    }

    #[test]
    #[should_panic]
    fn test_panic_on_productions_and_attractions_mismatched_lengths() {
        let max_iter = 25000;
        let eps = 1e-8;
    
        let mut productions_vector = array![100., 200., 100.];
        let mut attractions_vector = array![150., 150.];

        let peoples_num = productions_vector.sum();

        productions_vector /= peoples_num;
        attractions_vector /= peoples_num; 

        let cost_matrix = array![
            [1., 2.],
            [2., 1.]];

        sinkhorn(&productions_vector, &attractions_vector, &cost_matrix, max_iter, eps, peoples_num as i32);
    }

    #[test]
    fn test_peoples_num_equal_to_correspondence_sum() {
        let max_iter = 25000;
        let eps = 1e-8;

        let mut productions_vector = array![100., 200.];
        let mut attractions_vector = array![150., 150.];

        let peoples_num = productions_vector.sum();

        productions_vector /= peoples_num;
        attractions_vector /= peoples_num;

        let cost_matrix = array![
            [1., 2.],
            [2., 1.]];

        let (correspondence, _, _) = sinkhorn(&productions_vector, &attractions_vector, &cost_matrix, max_iter, eps, peoples_num as i32);
        
        assert_eq!(correspondence.sum(), peoples_num);
    }
}