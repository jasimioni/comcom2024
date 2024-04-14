# Create user

[Docs](https://www.rabbitmq.com/docs/access-control)

```
echo "remote" | rabbitmqctl add_user "remote"
rabbitmqctl set_permissions -p "/" "remote" ".*" ".*" ".*"
```

