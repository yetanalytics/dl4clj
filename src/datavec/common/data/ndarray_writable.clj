(ns ^{:doc "see: https://deeplearning4j.org/datavecdoc/org/datavec/common/data/NDArrayWritable.html"}
    datavec.common.data.ndarray-writable
  (:import [org.datavec.common.data NDArrayWritable]))

(defn new-ndarray-writable
  "A Writable that basically wraps an INDArray.

  :array (INDArray), an array from nd4j
   - if not supplied, will need to use set-array!"
  [& {:keys [array]
      :as opts}]
  (if (contains? opts :array)
    (NDArrayWritable. array)
    (NDArrayWritable.)))

(defn get-array
  "returns the INDArray from the writable"
  [array-writable]
  (.get array-writable))

(defn get-array-length
  "returns the length of the array"
  [array-writable]
  (.length array-writable))

(defn set-array!
  "sets the array for the ndarray writable object and returns the object"
  [& {:keys [array-writable ind-array]}]
  (doto array-writable (.set ind-array)))

(defn read-fields-into-array!
  "Deserialize into a row vector of default type."
  [& {:keys [in array-writable]}]
  (with-open [i in]
    (doto array-writable (.readFields i))))

(defn write-fields-from-array!
  "Serialize array data linearly."
  [& {:keys [array-writable out]}]
  (with-open [o out]
    (doto array-writable (.write o))))
