function ajaxpost () {
  // (A) GET FORM DATA
  var form = document.getElementById("myForm");
  var object = {};
  formData.forEach((value, key) => object[key] = value);
  var json = JSON.stringify(object);
 
  // (B) AJAX REQUEST
  // (B1) POST DATA TO SERVER, RETURN RESPONSE AS TEXT
  fetch("https://vyfeovv08c.execute-api.us-east-2.amazonaws.com/beta", { method:"POST", body:json})
  .then(res=>res.text())
 
  // (B2) SHOW MESSAGE ON SERVER RESPONSE
  .then((response) => {
    console.log(response);
    if (response == "Low Risk") { document.getElementById("strokeRiskResult").appendChild(<iframe src="https://giphy.com/embed/ZgCM9TWQ5lL8zpc4G2" width="480" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/ZgCM9TWQ5lL8zpc4G2">via GIPHY</a></p>) ;}
    else { document.getElementById("strokeRiskResult").appendChild(<iframe src="https://giphy.com/embed/ZgCM9TWQ5lL8zpc4G2" width="480" height="480" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/ZgCM9TWQ5lL8zpc4G2">via GIPHY</a></p>) ; }
  })
 
  // (B3) OPTIONAL - HANDLE FETCH ERROR
  .catch((err) => { console.error(err); });
 
  // (C) PREVENT FORM SUBMIT
  return false;
}
