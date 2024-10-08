ansible-playbook -i deployment/inventory deployment/site.yml --extra-vars "trustme=yes"


## Deployment

To deploy the project on a virtual server, configure the destination host in `deployment/inventory`. For example, if you want to deploy on a virtual machine which forwards the guest port 22 to the host 8022, use the following configuration:

```bash
target ansible_host=127.0.0.1 ansible_port=8022 ansible_user=root
```

Next up, configure the variables at the top of `deployment/site.yml`. On your workstation, install the dependency for deployment from `requirements/deployment.txt`, then launch Ansible with the following command:

```bash
ansible-playbook -i deployment/inventory deployment/site.yml
```

To deploy Nginx with a fake SSL certificate for `decoupled-django.com` and `static.decoupled-django.com`, run:

```bash
ansible-playbook -i deployment/inventory deployment/site.yml --extra-vars "trustme=yes"
```

If you want to skip the SSL part, run:

```bash
ansible-playbook -i deployment/inventory deployment/site.yml --skip-tags=ssl
```

