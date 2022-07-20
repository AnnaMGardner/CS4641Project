function ajaxpost () {
  // (A) GET FORM DATA
  var form = document.getElementById("myForm");
  var object = {};
  var link = "https://giphy.com/embed/ZgCM9TWQ5lL8zpc4G2";
  var iframe = document.createElement('iframe');
  iframe.frameBorder=0;
  iframe.width="300px";
  iframe.height="250px";
  iframe.id="randomid";
  iframe.setAttribute("src", link);
  var formData = new FormData(document.querySelector('form'))
  var formDataObject = Object.fromEntries(formData.entries());
  // Format the plain form data as JSON
  var formDataJsonString = JSON.stringify(formDataObject);
  console.log(formDataJsonString)
  // var formData = new FormData(document.querySelector('form'))
  // formData.forEach((value, key) => object[key] = value);
  // var json = JSON.stringify(object);
  // console.log(json)
 
  // (B) AJAX REQUEST
  // (B1) POST DATA TO SERVER, RETURN RESPONSE AS TEXT
  fetch("https://vyfeovv08c.execute-api.us-east-2.amazonaws.com/beta", { method:"POST", headers: {
    'Content-Type': 'application/json;charset=utf-8'
  }, body:formDataJsonString})
  .then((response) => {
    console.log("got here")
    if (response.text() == "Low Risk") { document.getElementById("strokeRiskResult").appendChild(iframe) ;}
    else { document.getElementById("strokeRiskResult").appendChild(iframe) ; }
  })
 
  // (B3) OPTIONAL - HANDLE FETCH ERROR
  .catch((err) => { console.error(err); });
 
  // (C) PREVENT FORM SUBMIT
  return false;
}
