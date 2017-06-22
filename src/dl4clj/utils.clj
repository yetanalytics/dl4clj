(ns dl4clj.utils
  (:import [org.deeplearning4j.util ModelSerializer]))

(defn contains-many? [m & ks]
  (every? #(contains? m %) ks))

(defn get-labels
  "returns labels for various types of objects in dl4j"
  [this]
  (.getLabels this))

(defn generic-dispatching-fn
  [opts]
  (first (keys opts)))

(defn camelize
  "Turn a symbol or keyword or string to a camel-case verion, e.g. (camelize :foo-bar) => :FooBar"
  [x & capitalize?]
  (let [parts (clojure.string/split (name x) #"[\s_-]+")
        not-capitalized (clojure.string/join "" (cons (first parts)
                                                      (map #(str (clojure.string/upper-case (subs % 0 1)) (subs % 1))
                                                           (rest parts))))
        new-name (if capitalize?
                   (clojure.string/join [(clojure.string/upper-case (subs not-capitalized 0 1))
                                         (subs not-capitalized 1)])
                   not-capitalized)]
    (condp = (type x)
      java.lang.String new-name
      clojure.lang.Keyword (keyword new-name)
      clojure.lang.Symbol (symbol new-name))))

(defn camel-to-dashed
  "Turn a symbol or keyword or string like 'bigBlueCar' to 'big-blue-car'."
  [x & capitalize?]
  (let [parts (or (re-seq #"[a-zA-Z][A-Z\s_]*[^A-Z\s_]*" (name x))
                  [(name x)])
        new-name (clojure.string/join "-" (map clojure.string/lower-case parts))]
    (condp = (type x)
      java.lang.String new-name
      clojure.lang.Keyword (keyword new-name)
      clojure.lang.Symbol (symbol new-name))))

(defn indexed [col]
  (map vector col (range)))

(defn array-of
  "takes in a data structure and a java class type.

  puts the data structure into an array with the java class as the type"
  [& {:keys [data java-type]}]
  (if (or (seq? data) (vector? data))
    (into-array java-type data)
    (into-array java-type [data])))

(defn save-model!
  "saves a model to a file or output stream

  :model (model), the multi layer network you want to save

  :path (str), path to the file you want to save the model in
   - recomended file extension is .bin

  :out (output stream), an output stream to save the model to

  :save-updater? (boolean), do we save the state of the updater?
   - defaults to true

  :save-normalizer? (boolean), do we want to save the normalizer used on the dataset?

  :normalizer (ds-pre-processor), a pre-processor that was applied to the training/testing datasets"
  [& {:keys [model path save-normalizer? out save-updater? normalizer]
      :or {save-updater? true}
      :as opts}]
  (let [f (clojure.java.io/as-file path)]
   (cond (contains-many? opts :model :path :save-normalizer? :normalizer)
         (do (ModelSerializer/writeModel model f save-updater?)
             (ModelSerializer/addNormalizerToModel f normalizer)
            model)
        (contains-many? opts :model :path)
        (do (ModelSerializer/writeModel model f save-updater?)
            model)
        (contains-many? opts :model :out)
        (ModelSerializer/writeModel model out save-updater?)
        :else
        (assert (or
                 (contains-many? opts :model :out)
                 (contains-many? opts :model :path))
                "you must supply a model and a place to save it"))))

(defn load-model!
  "loads a model from a file or input stream

  :path (str), path to the file in which the model is saved

  :in (input stream), the input stream containing the saved model

  :load-updater? (boolean), do we want to load the updater for the model
   - the model must have been saved with :save-updater? set to true

  :load-normalizer? (boolean), do we want to load the normalizer used on the
   training/testing datasets
    - the model must have been saved with :save-normalizer? set to true and the
      normalizer supplied to save-model!"
  [& {:keys [path in load-updater? load-normalizer?]
      :as opts}]
  (let [f (clojure.java.io/as-file path)]
   (cond (contains-many? opts :path :load-updater? :load-normalizer?)
         {:model (ModelSerializer/restoreMultiLayerNetwork f load-updater?)
          :normalizer (ModelSerializer/restoreNormalizerFromFile f)}
         (contains-many? opts :path :load-updater?)
         (ModelSerializer/restoreMultiLayerNetwork f load-updater?)
         (contains-many? opts :in :load-updater?)
         (ModelSerializer/restoreMultiLayerNetwork in load-updater?)
         (contains? opts :path)
         (ModelSerializer/restoreMultiLayerNetwork path)
         (contains? opts :in)
         (ModelSerializer/restoreMultiLayerNetwork in)
         :else
         (assert (or (contains? opts :path)
                     (contains? opts :in))
                 "you must supply a source to load the model from"))))
