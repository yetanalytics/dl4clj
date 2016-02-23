(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/api/ndarray/INDArray.html"}
  nd4clj.linalg.api.ndarray.indarray
  (:refer-clojure :exclude [min max])
  (:import [org.nd4j.linalg.api.ndarray INDArray]))

(defn put-scalar 
  "Insert the item at the specified indices"
  [^INDArray a indices value]  
  (.putScalar a (int-array indices) value))

(defn put-row
  "Insert a row in to this array Will throw an exception if this ndarray is not a matrix"
  [^INDArray a row to-put]
  (.putRow a (int row) to-put))

(defn get-scalar 
  "Get the vector along a particular dimension"
  [^INDArray a indices]
  (.getScalar a (int-array indices)))

(defn shape
  "Returns the shape of an ndarray"
   [^INDArray a]
  (into [] (.shape a)))

(defn data 
  "Returns a linear double array representation of this ndarray"
  [^INDArray a]
  (let [da (double-array (.length a))]
    (dotimes [i (.length a)]
      (aset da i (.getDouble a (int i))))
    (into [] da)))

(defn slice
  "Returns the specified slice of this ndarray"
  ([^INDArray a i]
   (.slice a (int i)))
  ([^INDArray a i dimension]
   (.slice a (int i) (int dimension))))

(defn max
  "Returns the overall max of this ndarray"
  [^INDArray this dimension]
  (.max this (int-array [dimension])))

(defn mean
  "Returns the overall mean of this ndarray"
  [^INDArray this dimension]
  (.mean this (int-array [dimension])))

(defn min
  "Returns the overall mean of this ndarray"
  [^INDArray this dimension]
  (.min this (int-array [dimension])))





(defn tensor-along-dimension [^INDArray a index dimensions]
  (.tensorAlongDimension a (int index) (int-array dimensions)))

(defn get-double [^INDArray a indices]
  (.getDouble a (int-array indices)))


(comment 
  
  (def a (nd4clj.linalg.factory.nd4j/create [1 2 3 4] [2 2]))
  (data a)
  
  )
