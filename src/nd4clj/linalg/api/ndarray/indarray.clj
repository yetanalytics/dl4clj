(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/api/ndarray/INDArray.html"}
  nd4clj.linalg.api.ndarray.indarray
  (:import [org.nd4j.linalg.api.ndarray INDArray]))

(defn put-scalar [^INDArray a indices ^Number value]
  (.putScalar a (int-array indices) value))

(defn get-scalar [^INDArray a indices]
  (.getScalar a (int-array indices)))

(defn get-double [^INDArray a indices]
  (.getDouble a (int-array indices)))

(defn shape [^INDArray a]
  (into [] (.shape a)))

(defn tensor-along-dimension [^INDArray a index dimensions]
  (.tensorAlongDimension a (int index) (int-array dimensions)))

(defn data [^INDArray a]
  (into [] (.data a)))

(defn slice 
  ([^INDArray a i]
   (.slice a (int i)))
  ([^INDArray a i dimension]
   (.slice a (int i) (int dimension))))

