(ns ^{:doc "see http://nd4j.org/apidocs/org/nd4j/linalg/factory/Nd4j.html"}
  nd4clj.linalg.factory.nd4j
  (:import [org.nd4j.linalg.factory Nd4j]))
;; (remove-ns 'nd4clj.linalg.factory.nd4j)

(defn zeros 
  ([shape]
   (Nd4j/zeros (int-array shape)))
  ([rows columns]
   (Nd4j/zeros (int rows) (int columns))))


(defmulti create-from-data (fn [data & more] (mapv type more)))
(defmethod create-from-data []
  [data]
  (Nd4j/create (double-array data)))
(defmethod create-from-data [java.lang.Character]
  [data order]
  (Nd4j/create (double-array data) ^java.lang.Character order))
(defmethod create-from-data [clojure.lang.IPersistentCollection]
  [data shape]
  (Nd4j/create (double-array data) (int-array shape)))
(defmethod create-from-data [clojure.lang.IPersistentCollection java.lang.Character]
  [data shape ordering]
  (Nd4j/create (double-array data) (int-array shape) ^java.lang.Character ordering))
(defmethod create-from-data [clojure.lang.IPersistentCollection Number]
  [data shape offset]
  (Nd4j/create (double-array data) (int-array shape) (int offset)))
(defmethod create-from-data [clojure.lang.IPersistentCollection clojure.lang.IPersistentCollection Number]
  [data shape stride offset]
  (Nd4j/create (double-array data) (int-array shape) (int-array stride) (int offset)))
(defmethod create-from-data [clojure.lang.IPersistentCollection clojure.lang.IPersistentCollection Number java.lang.Character]
  [data shape stride offset ordering]
  (Nd4j/create (double-array data) (int-array shape) (int-array stride) (int offset) ^java.lang.Character ordering))
(defmethod create-from-data [clojure.lang.IPersistentCollection Number java.lang.Character]
  [data shape offset ordering]
  (Nd4j/create (double-array data) (int-array shape) (int offset) ^java.lang.Character ordering))
(defmethod create-from-data [clojure.lang.IPersistentCollection Number Number clojure.lang.IPersistentCollection Number]  
  [data rows columns stride offset]
  (Nd4j/create (double-array data) (int rows) (int columns) (int-array stride) (int offset)))
(defmethod create-from-data [clojure.lang.IPersistentCollection Number Number clojure.lang.IPersistentCollection Number java.lang.Character]
  [data rows columns stride offset ordering]
  (Nd4j/create (double-array data) (int rows) (int columns) (int-array stride) (int offset) ^java.lang.Character ordering))
(defmethod create-from-data [clojure.lang.IPersistentCollection Number Number clojure.lang.IPersistentCollection Number java.lang.Character]
  [data rows columns stride offset ordering]
  (Nd4j/create (double-array data) (int rows) (int columns) (int-array stride) (int offset) ^java.lang.Character ordering))


(defmulti create-from-shape (fn [& more] (mapv type more)))
(defmethod create-from-shape [Number]
  [columns]
  (Nd4j/create (int columns)))

(defmethod create-from-shape [Number java.lang.Character]
  [columns ordering]
  (Nd4j/create (int columns) ^java.lang.Character ordering))
(defmethod create-from-shape [Number Number]
  [rows columns]
  (Nd4j/create (int rows) (int columns)))
(defmethod create-from-shape [Number Number java.lang.Character]
  [rows columns order]
  (Nd4j/create (int rows) (int columns) ^java.lang.Character order))
(defmethod create-from-shape [Number Number clojure.lang.IPersistentCollection]
  [rows columns stride]
  (Nd4j/create (int rows) (int columns) (int-array stride)))
(defmethod create-from-shape [Number Number clojure.lang.IPersistentCollection java.lang.Character]
  [rows columns stride ordering]
  (Nd4j/create (int rows) (int columns) (int-array stride) ^java.lang.Character ordering))
(defmethod create-from-shape [Number Number clojure.lang.IPersistentCollection Number]
  [rows columns stride offset]
  (Nd4j/create (int rows) (int columns) (int-array stride) (int offset)))
(defmethod create-from-shape [Number Number clojure.lang.IPersistentCollection Number java.lang.Character]
  [rows columns stride offset ordering]
  (Nd4j/create (int rows) (int columns) (int-array stride) (int offset) ^java.lang.Character ordering))
(defmethod create-from-shape [clojure.lang.IPersistentCollection]
  [shape]
  (Nd4j/create (int-array shape)))
(defmethod create-from-shape [clojure.lang.IPersistentCollection java.lang.Character]
  [shape ordering]
  (Nd4j/create (int-array shape) ^java.lang.Character ordering))
(defmethod create-from-shape [clojure.lang.IPersistentCollection clojure.lang.IPersistentCollection]
  [shape stride]
  (Nd4j/create (int-array shape) (int-array stride)))
(defmethod create-from-shape [clojure.lang.IPersistentCollection clojure.lang.IPersistentCollection java.lang.Character]
  [shape stride ordering]
  (Nd4j/create (int-array shape) (int-array stride) ^java.lang.Character ordering))
(defmethod create-from-shape [clojure.lang.IPersistentCollection clojure.lang.IPersistentCollection Number]
  [shape stride offset]
  (Nd4j/create (int-array shape) (int-array stride) (int offset)))
(defmethod create-from-shape [clojure.lang.IPersistentCollection clojure.lang.IPersistentCollection Number java.lang.Character]
  [shape stride offset ordering]
  (Nd4j/create (int-array shape) (int-array stride) (int offset) ^java.lang.Character ordering))



;; (defn create [data shape]
;;   (Nd4j/create (double-array data) (int-array shape)))

(defn enforce-numerical-stability? []
  (Nd4j/ENFORCE_NUMERICAL_STABILITY))

(defn set-enforce-numerical-stability! [^Boolean value]
  (set! (Nd4j/ENFORCE_NUMERICAL_STABILITY) value))


