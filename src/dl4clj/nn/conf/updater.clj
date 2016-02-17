(ns ^{:doc "see http://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/Updater.html"}
  dl4clj.nn.conf.updater
  (:import [org.deeplearning4j.nn.conf Updater]))

(defn value-of [k]
  (if (string? k)
    (Updater/valueOf k)
    (Updater/valueOf (clojure.string/replace (clojure.string/upper-case (name k)) "-" "_"))))

(defn values
  "Returns collection of supported updaters:
  (:sgd :adam :adadelta :nesterovs :adagrad :rmsprop)
  "
  []
  (map #(keyword (clojure.string/replace (clojure.string/lower-case (str %)) "_" "-")) (Updater/values)))

(comment
  
  (values)
  (value-of :adagrad)
  (value-of "ADAGRAD")
  
)
