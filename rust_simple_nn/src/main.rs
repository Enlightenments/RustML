use std::{error::Error, process::Output};

use ndarray::{array, Array1, Array2};
use rand::{thread_rng, Rng};

/// 简单全连接神经网络  

#[derive(Debug)]
pub struct NN {
    pub w1: f64,
    pub w2: f64,
    pub w3: f64,
    pub w4: f64,
    pub w5: f64,
    pub w6: f64,
    //
    pub b1: f64,
    pub b2: f64,
    pub b3: f64,
}

impl NN {
    pub fn new() -> Self {
        let mut rng = thread_rng();
        let nn = Self {
            w1: rng.gen(),
            w2: rng.gen(),
            w3: rng.gen(),
            w4: rng.gen(),
            w5: rng.gen(),
            w6: rng.gen(),
            b1: rng.gen(),
            b2: rng.gen(),
            b3: rng.gen(),
        };
        nn
    }

    /// sigmoid = 1 / ( 1 + e^-x )
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    /// d_sigmoid = sigmoid(x) * ( 1 - sigmoid(x) )
    pub fn d_sigmoid(x: f64) -> f64 {
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }
    /// mean-square error, y = (E(y(i)-y(i)')^2)/n
    pub fn mse(actual: &Array1<f64>, predict: &Array1<f64>) -> f64 {
        if actual.len() == predict.len() && actual.len() > 0 {
            let mut sum = 0.0;
            for i in 0..actual.len() {
                sum = sum + (actual[i] - predict[i]) * (actual[i] - predict[i]);
            }
            sum / (actual.len() as f64)
        } else {
            -1.0
        }
    }
    /// forward
    pub fn forward(&self, x: &Array1<f64>) -> (f64, f64, f64, f64, f64, f64) {
        let h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1;
        let h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2;
        let h1f = Self::sigmoid(h1);
        let h2f = Self::sigmoid(h2);
        let o1 = self.w5 * h1f + self.w6 * h2f + self.b3;
        let o1f = Self::sigmoid(o1);
        (h1, h1f, h2, h2f, o1, o1f)
    }

    /// backword
    pub fn backword(&mut self, data: Array2<f64>, y_trues: &Array1<f64>, times: usize) {
        let learn_rate = 0.01;
        for index in 1..=times {
            for (x, y_true) in data.rows().into_iter().zip(y_trues.iter()) {
                let (h1, h1f, h2, h2f, o1, o1f) = self.forward(&x.to_owned());
                //
                let dpre = -1. * (y_true - o1f);
                //do1 / dh1
                let dh1 = self.w5 * Self::d_sigmoid(o1);
                let dh2 = self.w6 * Self::d_sigmoid(o1);
                //
                let dw5 = h1 * Self::d_sigmoid(o1);
                let dw6 = h2 * Self::d_sigmoid(o1);
                let db3 = Self::d_sigmoid(o1);
                // dh1 / dw
                let dh1_dw1 = x[0] * Self::d_sigmoid(h1);
                let dh1_dw2 = x[1] * Self::d_sigmoid(h1);
                let db1 = Self::d_sigmoid(h1);
                // dh2 / dw
                let dh2_dw1 = x[0] * Self::d_sigmoid(h2);
                let dh2_dw2 = x[1] * Self::d_sigmoid(h2);
                let db2 = Self::d_sigmoid(h2);
                //
                //update w b
                self.w1 -= learn_rate * dpre * dh1 * dh1_dw1;
                self.w2 -= learn_rate * dpre * dh1 * dh1_dw2;
                self.b1 -= learn_rate * dpre * dh1 * db1;
                self.w3 -= learn_rate * dpre * dh2 * dh2_dw1;
                self.w4 -= learn_rate * dpre * dh2 * dh2_dw2;
                self.b2 -= learn_rate * dpre * dh2 * db2;
                self.w5 -= learn_rate * dpre * dw5;
                self.w6 -= learn_rate * dpre * dw6;
                self.b3 -= learn_rate * dpre * db3;
            }
            if index % 10000 == 0 {
                let mut y_preds = Array1::<f64>::zeros(data.shape()[0]);
                for (i, x) in data.rows().into_iter().enumerate() {
                    y_preds[i] = self.forward(&x.to_owned()).5;
                }

                let loss = Self::mse(y_trues, &y_preds);
                println!("index: {}, loss: {}", index, loss)
            }
        }
    }
}

fn main() {
    println!("Hello, world!");
    //
    let data = array![
        [-2., -1.],
        [25., 6.],
        [17., 4.],
        [-15., -6.],
        [-16., -6.],
        [-14., -6.],
    ];
    let y_trues = array![1., 0., 0., 1., 1., 1.];
    //
    let mut nn = NN::new();
    print!("nn:{:?}", nn);
    nn.backword(data, &y_trues, 100000);
    print!("{:?}",nn);
}
