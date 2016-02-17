(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/factory/Nd4j.html"}
  nd4clj.linalg.factory.nd4j
  (:import [org.nd4j.linalg.factory Nd4j]))

(defn zeros 
  ([shape]
   (Nd4j/zeros (int-array shape)))
  ([rows columns]
   (Nd4j/zeros (int rows) (int columns))))

(defn create [data shape]
  (Nd4j/create (double-array data) (int-array shape)))

(defn enforce-numerical-stability? []
  (Nd4j/ENFORCE_NUMERICAL_STABILITY))

(defn set-enforce-numerical-stability! [^Boolean value]
  (set! (Nd4j/ENFORCE_NUMERICAL_STABILITY) value))


