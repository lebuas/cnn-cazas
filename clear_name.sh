BASURA=""

while read name; do
  # 1. Extraemos el nombre base
  base=$(echo "$name" | cut -d'_' -f1)

  # 2. Buscamos si la base ya está en nuestra variable "global"
  if [[ "$BASURA" == *"$base"* ]]; then
    # Si ya está, lo mandamos al txt (con >> para no borrar lo anterior)
    echo "$name" >>basura.txt
  else
    # Si no está, lo sumamos a la variable
    BASURA="$BASURA $base"
  fi

done <<<"$(ls)"

echo "Proceso terminado"
