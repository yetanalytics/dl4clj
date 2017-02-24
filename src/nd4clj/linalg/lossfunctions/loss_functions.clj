(ns ^{:doc "see http://nd4j.org/doc/org/nd4j/linalg/lossfunctions/LossFunctions.html
            and http://nd4j.org/doc/org/nd4j/linalg/lossfunctions/LossFunctions.LossFunction.html"}
  nd4clj.linalg.lossfunctions.loss-functions
  (:import [org.nd4j.linalg.lossfunctions LossFunctions LossFunctions$LossFunction])
  (:require [clojure.string :as s]))


(defn value-of [k]
  (if (string? k)
    (LossFunctions$LossFunction/valueOf k)
    (LossFunctions$LossFunction/valueOf (s/replace (s/upper-case (name k)) "-" "_"))))

(defn values
  "List supported loss functions.

  :mse Mean Squared Error (Linear Regression)
  :expll Exponential log likelihood (Poisson Regression)
  :xent Cross Entropy (Binary Classification)
  :softmax Softmax Regression
  :rmse-xent RMSE Cross Entropy
  :expll
  :mcxent
  :squared-loss
  :reconstruction-crossentropy
  :negativeloglikelihood"
  []
  (map #(keyword (s/replace (s/lower-case (str %)) "_" "-")) (LossFunctions$LossFunction/values)))

(comment

  (map value-of (values))
  (value-of :mse)
  (value-of :l1)
  (value-of :expll) ;; deprecated
  (value-of :xent)
  (value-of :mcxent)
  (value-of :rmse-xent) ;; deprecated
  (value-of :squared-loss)
  (value-of :reconstruction-crossentropy)
  (value-of :negativeloglikelihood)
  (value-of :custom) ;; deprecated
  (value-of :cosine-proximity)
  (value-of :hinge)
  (value-of :squared-hinge)
  (value-of :kl-divergence)
  (value-of :mean-absolute-error)
  (value-of :l2)
  (value-of :mean-absolute-percentage-error)
  (value-of :mean-squared-logarithmic-error)
  (value-of :poisson)
  (type (value-of :reconstruction-crossentropy))
  )
